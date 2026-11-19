"""
Prepare the nampdn-ai/tiny-webtext dataset and train GPT-2 with Multi-Head Latent Attention.

  # 1) download + tokenize once (writes data/train.bin and data/val.bin)
  python train.py --prepare                     # full dataset
  python train.py --prepare --max_docs 20000    # small subset for quick tests

  # 2) train (all GPTConfig / TrainConfig fields can be passed as --flags)
  python train.py
  python train.py --n_layer 12 --n_head 12 --n_embd 768 --kv_lora_rank 256 --batch_size 16 --grad_accum_steps 4
  python train.py --attn_type mha --out_dir out_mha          # standard-attention baseline
  python train.py --resume                                    # continue from out_dir/ckpt_last.pt
"""

import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch

from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    # data
    dataset: str = "nampdn-ai/tiny-webtext"
    data_dir: str = "data"
    text_column: str = "human"    # tiny-webtext: "human" = web text, "bot" = AI commentary; "human,bot" = both
    max_docs: int = 0             # 0 = use the whole dataset (only used by --prepare)
    val_frac: float = 0.005       # used if the dataset has no validation split
    num_proc: int = max(1, (os.cpu_count() or 2) // 2)
    # i/o
    out_dir: str = "out"
    resume: bool = False
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    sample_tokens: int = 64       # tokens of sample text printed at each eval (0 = off)
    # optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_iters: int = 5000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # system
    device: str = "auto"          # auto / cuda / mps / cpu
    dtype: str = "auto"           # auto / bfloat16 / float16 / float32
    compile: bool = False
    seed: int = 1337


# --------------------------------------------------------------------------------------
# Data preparation: HF dataset -> uint16 token files read with np.memmap
# --------------------------------------------------------------------------------------
def prepare_data(tc: TrainConfig):
    from datasets import load_dataset, DatasetDict
    import tiktoken
    from tqdm import tqdm

    os.makedirs(tc.data_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(tc.data_dir, f"{s}.bin")) for s in ("train", "val")):
        print(f"{tc.data_dir}/train.bin and val.bin already exist; delete them to re-prepare.")
        return

    ds = load_dataset(tc.dataset)
    print(ds)
    train = ds["train"]
    val = ds.get("validation") or ds.get("test")
    if tc.max_docs:
        train = train.select(range(min(tc.max_docs, len(train))))
    if val is None:
        split = train.train_test_split(test_size=tc.val_frac, seed=tc.seed, shuffle=True)
        train, val = split["train"], split["test"]
    elif tc.max_docs:
        val = val.select(range(min(max(1, int(tc.max_docs * tc.val_frac)) + 100, len(val))))

    # pick the text column(s); fall back to the first string column if not present
    text_cols = [c for c in tc.text_column.split(",") if c in train.column_names]
    if not text_cols:
        text_cols = [next(n for n, f in train.features.items()
                          if getattr(f, "dtype", None) in ("string", "large_string"))]
    print(f"using text column(s): {text_cols}")

    enc = tiktoken.get_encoding("gpt2")

    def tokenize(batch):
        texts = ["\n\n".join(parts) for parts in zip(*(batch[c] for c in text_cols))]
        ids = enc.encode_ordinary_batch(texts)
        ids = [x + [enc.eot_token] for x in ids]  # end-of-text token separates documents
        return {"ids": ids, "len": [len(x) for x in ids]}

    for name, split in DatasetDict(train=train, val=val).items():
        tok = split.map(tokenize, batched=True, remove_columns=split.column_names,
                        num_proc=tc.num_proc, desc=f"tokenizing {name}")
        total = int(np.sum(tok["len"], dtype=np.uint64))
        path = os.path.join(tc.data_dir, f"{name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        n_shards = min(1024, max(1, len(tok) // 1000))
        pos = 0
        for i in tqdm(range(n_shards), desc=f"writing {path}"):
            shard = tok.shard(num_shards=n_shards, index=i, contiguous=True).with_format("numpy")
            chunk = np.concatenate(shard["ids"]) if len(shard) else np.array([], dtype=np.uint16)
            arr[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        arr.flush()
        print(f"{name}: {len(tok):,} docs, {total:,} tokens -> {path}")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def get_batch(data_dir, split, batch_size, block_size, device):
    # re-open the memmap every call to avoid a memory leak (see nanoGPT)
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(it, tc: TrainConfig):
    """Linear warmup followed by cosine decay to min_lr."""
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / (tc.warmup_iters + 1)
    if it >= tc.max_iters:
        return tc.min_lr
    ratio = (it - tc.warmup_iters) / max(1, tc.max_iters - tc.warmup_iters)
    return tc.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (tc.learning_rate - tc.min_lr)


@torch.no_grad()
def estimate_loss(model, tc, mc, device, ctx):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            x, y = get_batch(tc.data_dir, split, tc.batch_size, mc.block_size, device)
            with ctx:
                _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, mc, tc, it, best_val):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": asdict(mc), "train_config": asdict(tc),
                "iter": it, "best_val_loss": best_val}, path)


def train(mc: GPTConfig, tc: TrainConfig):
    # ---- device / precision ----
    if tc.device == "auto":
        tc.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device_type = "cuda" if tc.device.startswith("cuda") else tc.device
    if tc.dtype == "auto":
        tc.dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device_type == "cuda" else "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tc.dtype]
    ctx = nullcontext() if ptdtype == torch.float32 else torch.autocast(device_type=device_type, dtype=ptdtype)
    torch.manual_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(tc.out_dir, exist_ok=True)

    for s in ("train", "val"):
        if not os.path.exists(os.path.join(tc.data_dir, f"{s}.bin")):
            raise SystemExit(f"{tc.data_dir}/{s}.bin not found -- run `python train.py --prepare` first.")

    # ---- model / optimizer (fresh or resumed) ----
    start_iter, best_val = 0, float("inf")
    ckpt_path = os.path.join(tc.out_dir, "ckpt_last.pt")
    if tc.resume:
        model, ckpt = GPT.from_checkpoint(ckpt_path, tc.device)
        mc = model.config
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val_loss"]
        print(f"resumed from {ckpt_path} at iter {ckpt['iter']} (best val {best_val:.4f})")
    else:
        model = GPT(mc).to(tc.device)
    optimizer = model.configure_optimizer(tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2), device_type)
    if tc.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt

    print(f"device={tc.device} dtype={tc.dtype} attn={mc.attn_type}  "
          f"params={model.num_params() / 1e6:.2f}M (non-embedding)  "
          f"KV-cache/token={model.kv_cache_elems_per_token()} elems "
          f"(MHA equivalent {model.mha_equivalent_kv_cache_elems_per_token()})")
    print(f"tokens per iteration: {tc.batch_size * tc.grad_accum_steps * mc.block_size:,}")

    raw_model = model
    if tc.compile:
        print("compiling the model...")
        model = torch.compile(model)
    scaler = torch.amp.GradScaler(enabled=(tc.dtype == "float16"))

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    log_path = os.path.join(tc.out_dir, "log.csv")
    if not tc.resume or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "train_loss", "val_loss", "lr"])

    # ---- training loop ----
    x, y = get_batch(tc.data_dir, "train", tc.batch_size, mc.block_size, tc.device)
    t0 = time.time()
    for it in range(start_iter, tc.max_iters + 1):
        lr = get_lr(it, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # evaluation + checkpointing
        if it % tc.eval_interval == 0 and (it > start_iter or not tc.resume):
            losses = estimate_loss(model, tc, mc, tc.device, ctx)
            print(f"[eval] iter {it}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}"])
            if it > 0:
                save_checkpoint(ckpt_path, model, optimizer, mc, tc, it, min(best_val, losses["val"]))
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(os.path.join(tc.out_dir, "ckpt_best.pt"), model, optimizer, mc, tc, it, best_val)
                    print(f"  saved new best checkpoint (val {best_val:.4f})")
            if tc.sample_tokens:
                raw_model.eval()
                start = torch.tensor([[enc.eot_token]], device=tc.device)
                with ctx:
                    out = raw_model.generate(start, tc.sample_tokens, temperature=0.8, top_k=50)
                print("  sample:", repr(enc.decode(out[0, 1:].tolist())))
                raw_model.train()
        if it == tc.max_iters:
            break

        # forward / backward with gradient accumulation
        for _ in range(tc.grad_accum_steps):
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / tc.grad_accum_steps
            x, y = get_batch(tc.data_dir, "train", tc.batch_size, mc.block_size, tc.device)  # prefetch
            scaler.scale(loss).backward()
        if tc.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % tc.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = tc.batch_size * tc.grad_accum_steps * mc.block_size * (tc.log_interval if it > start_iter else 1) / dt
            print(f"iter {it}: loss {loss.item() * tc.grad_accum_steps:.4f}  lr {lr:.2e}  {tok_s:,.0f} tok/s")

    print(f"done. best val loss {best_val:.4f}; checkpoints in {tc.out_dir}/")


# --------------------------------------------------------------------------------------
# CLI: every field of GPTConfig and TrainConfig becomes a --flag
# --------------------------------------------------------------------------------------
def parse_args():
    def str2bool(v):
        return str(v).lower() in ("1", "true", "yes", "y")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", action="store_true", help="download + tokenize the dataset, then exit")
    for cls in (GPTConfig, TrainConfig):
        for f in fields(cls):
            if f.type in (bool, "bool"):
                p.add_argument(f"--{f.name}", type=str2bool, nargs="?", const=True, default=f.default)
            else:
                typ = {"int": int, "float": float, "str": str}.get(f.type, f.type)
                p.add_argument(f"--{f.name}", type=typ, default=f.default)
    args = vars(p.parse_args())
    mc = GPTConfig(**{f.name: args[f.name] for f in fields(GPTConfig)})
    tc = TrainConfig(**{f.name: args[f.name] for f in fields(TrainConfig)})
    return args["prepare"], mc, tc


if __name__ == "__main__":
    prepare, mc, tc = parse_args()
    if prepare:
        prepare_data(tc)
    else:
        train(mc, tc)
