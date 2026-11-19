"""
Generate text from a trained checkpoint.

  python generate.py --ckpt out/ckpt_best.pt --prompt "The best way to learn" --max_new_tokens 200
"""

import argparse
import time

import tiktoken
import torch

from model import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt_best.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--no_cache", action="store_true", help="disable the KV-cache (slower)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, ckpt = GPT.from_checkpoint(args.ckpt, args.device)
    model.eval()
    cfg = model.config

    mine = model.kv_cache_elems_per_token()
    mha = model.mha_equivalent_kv_cache_elems_per_token()
    print(f"loaded {args.ckpt} (iter {ckpt['iter']}, val loss {ckpt['best_val_loss']:.4f})")
    print(f"attn={cfg.attn_type}  params={model.num_params() / 1e6:.2f}M  "
          f"KV-cache per token: {mine} elems ({mine * 2 / 1024:.1f} KiB in fp16) "
          f"vs {mha} for MHA -> {mha / mine:.1f}x smaller")
    print("-" * 80)

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(args.prompt) if args.prompt else [enc.eot_token]
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    for _ in range(args.num_samples):
        t0 = time.time()
        out = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k, use_cache=not args.no_cache)
        dt = time.time() - t0
        text = enc.decode([t for t in out[0].tolist() if t < enc.n_vocab])
        print(text.replace("<|endoftext|>", "\n[EOT]\n"))
        print(f"--- {args.max_new_tokens / dt:.1f} tokens/s " + "-" * 50)


if __name__ == "__main__":
    main()
