"""
Generate text from trained checkpoints and compare the KV-cache memory of MLA and MHA.

  # compare mode (default when both out_mla/ckpt_best.pt and out_mha/ckpt_best.pt exist):
  python generate.py --prompt "The best way to learn" --max_new_tokens 200

  # pick the two checkpoints explicitly
  python generate.py --mla_ckpt out_mla/ckpt_best.pt --mha_ckpt out_mha/ckpt_best.pt

  # single model (old behaviour)
  python generate.py --ckpt out/ckpt_best.pt --prompt "The best way to learn"

In compare mode both models get the same prompt and the same random seed. During
generation the script measures the bytes held by the KV-cache tensors after every step,
prints a comparison table and saves plots/kv_cache_comparison.png (disable with --no_plot).
"""

import argparse
import os
import random
import time

import tiktoken
import torch

from model import GPT

MiB = 1024 ** 2


def run_model(path, args, enc, x, seed):
    """Load a checkpoint, generate num_samples samples and measure the KV-cache. Returns a stats dict."""
    model, ckpt = GPT.from_checkpoint(path, args.device)
    model.eval()
    cfg = model.config
    cuda = args.device.startswith("cuda")

    per_tok = model.kv_cache_elems_per_token()
    elem_size = next(model.parameters()).element_size()
    print(f"loaded {path} (iter {ckpt['iter']}, val loss {ckpt['best_val_loss']:.4f})")
    print(f"attn={cfg.attn_type}  params={model.num_params() / 1e6:.2f}M  "
          f"KV-cache per token: {per_tok} elems ({per_tok * elem_size / 1024:.2f} KiB at "
          f"{elem_size * 8}-bit, {per_tok * 2 / 1024:.2f} KiB in fp16)")
    print("-" * 80)

    cache_curve = []  # KV-cache bytes after each generated token (first sample)
    torch.manual_seed(seed)
    for s in range(args.num_samples):
        curve = []
        on_step = (lambda caches: curve.append(GPT.kv_cache_bytes(caches))) if not args.no_cache else None
        t0 = time.time()
        out = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k, use_cache=not args.no_cache, on_step=on_step)
        dt = time.time() - t0
        if s == 0:
            cache_curve = curve
        text = enc.decode([t for t in out[0].tolist() if t < enc.n_vocab])
        print(text.replace("<|endoftext|>", "\n[EOT]\n"))
        kv = f"  KV-cache peak {max(curve) / MiB:.3f} MiB" if curve else ""
        print(f"--- {args.max_new_tokens / dt:.1f} tokens/s{kv} " + "-" * 40)

    stats = dict(name=cfg.attn_type.upper(), path=path, cfg=cfg, per_tok_elems=per_tok,
                 elem_size=elem_size, curve=cache_curve, params=model.num_params())
    del model
    if cuda:
        torch.cuda.empty_cache()
    return stats


def print_comparison(results):
    print("\n" + "=" * 80)
    print("KV-cache memory during inference (batch size 1, measured from the cache tensors)")
    print("=" * 80)
    hdr = f"{'model':<6}{'elems/token':>13}{'KiB/token':>11}{'peak cache MiB':>16}{'full ctx MiB':>14}"
    print(hdr)
    for r in results:
        full = r["per_tok_elems"] * r["elem_size"] * r["cfg"].block_size / MiB
        peak = max(r["curve"]) / MiB if r["curve"] else float("nan")
        line = (f"{r['name']:<6}{r['per_tok_elems']:>13}{r['per_tok_elems'] * r['elem_size'] / 1024:>11.2f}"
                f"{peak:>16.3f}{full:>14.3f}")
        print(line)
    mla = next((r for r in results if r["name"] == "MLA"), None)
    mha = next((r for r in results if r["name"] == "MHA"), None)
    if mla and mha and mla["curve"] and mha["curve"]:
        print(f"\nMLA KV-cache is {max(mha['curve']) / max(mla['curve']):.2f}x smaller than MHA "
              f"(per token: {mha['per_tok_elems']} vs {mla['per_tok_elems']} elements)")
    print("peak cache = largest measured KV-cache;  full ctx = cache size with the whole block_size context filled")


def plot_comparison(results, path):
    import matplotlib.pyplot as plt
    from plot import SERIES, INK_2  # shared style (also applies plot.py's rcParams)

    fig, ax_c = plt.subplots(figsize=(9, 5))

    # measured cache size at every generation step
    for i, r in enumerate(results):
        c = SERIES[i % len(SERIES)]
        y = [b / MiB for b in r["curve"]]
        ax_c.plot(range(1, len(y) + 1), y, color=c, label=f"{r['name']} (peak {max(y):.2f} MiB)")
    ax_c.set(title="KV-cache size during generation (measured)", xlabel="generation step",
             ylabel="KV-cache (MiB)")
    ax_c.legend(loc="upper left")
    ax_c.text(0.99, 0.02, "drops = context window full, cache restarted on a sliding window",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=9, color=INK_2)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="generate from this single checkpoint only")
    p.add_argument("--mla_ckpt", default="out_mla/ckpt_best.pt")
    p.add_argument("--mha_ckpt", default="out_mha/ckpt_best.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--no_cache", action="store_true", help="disable the KV-cache (slower)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--plot", default="plots/kv_cache_comparison.png", help="where to save the comparison plot")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    # which checkpoints to run
    if args.ckpt:
        paths = [args.ckpt]
    else:
        paths = [c for c in (args.mla_ckpt, args.mha_ckpt) if os.path.exists(c)]
        if not paths:
            paths = ["out/ckpt_best.pt"]
    compare = len(paths) > 1

    # same seed for every model so the samples are comparable
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(args.prompt) if args.prompt else [enc.eot_token]
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    if compare:
        print(f"comparing {len(paths)} models: {', '.join(paths)}  (seed {seed})\n")

    results = []
    for path in paths:
        results.append(run_model(path, args, enc, x, seed))
        print()

    if compare:
        if args.no_cache:
            print("--no_cache given: no KV-cache to compare.")
            return
        print_comparison(results)
        if not args.no_plot:
            plot_comparison(results, args.plot)


if __name__ == "__main__":
    main()
