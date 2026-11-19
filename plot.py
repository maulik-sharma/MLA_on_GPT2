"""
Loss curves from training logs. Runs automatically at the end of train.py; can also be run by hand.

  python plot.py out_mla                    # plots for one run
  python plot.py out_mla out_mha            # plots for both runs + an MLA vs MHA comparison
  python plot.py out_mla --from_stdout train_output.txt   # rebuild train_log.csv from saved console output

Per run (saved in --plot_dir, default "plots/"):
  <run>_train_vs_val_loss.png     train and validation loss at every evaluation (from log.csv)
  <run>_train_loss_per_iter.png   training loss at every logged step, raw + smoothed (from train_log.csv)
With two or more runs:
  compare_loss.png                validation and smoothed training loss of all runs on shared axes
"""

import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# colours: fixed categorical order, recessive grid and axes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "font.size": 11, "lines.linewidth": 2,
})


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def ema(values, alpha=0.1):
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def run_name(out_dir):
    return os.path.basename(os.path.normpath(out_dir))


def label_end(ax, x, y, text, color):
    """Direct label at the right end of a line."""
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK_2)


def plot_train_val(out_dir, plot_dir):
    log = read_csv(os.path.join(out_dir, "log.csv"))
    if not log:
        return None
    name = run_name(out_dir)
    it, tr, va = log["iter"], log["train_loss"], log["val_loss"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(it, tr, color=SERIES[0], marker="o", markersize=4, label="train loss")
    ax.plot(it, va, color=SERIES[1], marker="s", markersize=4, label="validation loss")
    label_end(ax, it, tr, f"train {tr[-1]:.2f}", SERIES[0])
    label_end(ax, it, va, f"val {va[-1]:.2f}", SERIES[1])
    b = min(range(len(va)), key=va.__getitem__)
    ax.annotate(f"best val {va[b]:.4f}\nat iter {int(it[b])}", (it[b], va[b]), xytext=(0, 40),
                textcoords="offset points", ha="center", fontsize=10, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set(title=f"Train vs validation loss ({name})", xlabel="iteration", ylabel="cross-entropy loss")
    ax.legend(loc="upper right")
    ax.margins(x=0.08)
    path = os.path.join(plot_dir, f"{name}_train_vs_val_loss.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_train_steps(out_dir, plot_dir):
    log = read_csv(os.path.join(out_dir, "train_log.csv"))
    if not log:
        return None
    name = run_name(out_dir)
    it, loss = log["iter"], log["loss"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(it, loss, color=SERIES[0], linewidth=1, alpha=0.3, label="per step")
    ax.plot(it, ema(loss), color=SERIES[0], label="smoothed (EMA)")
    ax.set(title=f"Training loss per iteration ({name})", xlabel="iteration", ylabel="cross-entropy loss")
    ax.legend(loc="upper right")
    path = os.path.join(plot_dir, f"{name}_train_loss_per_iter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_compare(out_dirs, plot_dir):
    fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, d in enumerate(out_dirs):
        c, name = SERIES[i % len(SERIES)], run_name(d)
        log = read_csv(os.path.join(d, "log.csv"))
        if log:
            ax_v.plot(log["iter"], log["val_loss"], color=c, marker="o", markersize=3,
                      label=f"{name} (best {min(log['val_loss']):.4f})")
        steps = read_csv(os.path.join(d, "train_log.csv"))
        if steps:
            sm = ema(steps["loss"])
            ax_t.plot(steps["iter"], sm, color=c, label=f"{name} (final {sm[-1]:.2f})")
    ax_v.set(title="Validation loss", xlabel="iteration", ylabel="cross-entropy loss")
    ax_t.set(title="Training loss (smoothed)", xlabel="iteration")
    for ax in (ax_v, ax_t):
        ax.legend(loc="upper right")
        ax.margins(x=0.03)
    path = os.path.join(plot_dir, "compare_loss.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_run(out_dir, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    return [p for p in (plot_train_val(out_dir, plot_dir), plot_train_steps(out_dir, plot_dir)) if p]


def train_log_from_stdout(stdout_path, out_dir):
    """Rebuild train_log.csv from saved console output lines like 'iter 120: loss 6.53  lr 3.6e-04 ...'."""
    pat = re.compile(r"^iter (\d+): loss ([\d.]+)\s+lr ([\d.e+-]+)")
    rows = [m.groups() for line in open(stdout_path) if (m := pat.match(line.strip()))]
    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "loss", "lr"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_dir}/train_log.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dirs", nargs="+", help="training output directories (containing log.csv)")
    p.add_argument("--plot_dir", default="plots")
    p.add_argument("--from_stdout", help="saved console output of a training run (single out_dir only)")
    args = p.parse_args()
    if args.from_stdout:
        assert len(args.out_dirs) == 1, "--from_stdout needs exactly one out_dir"
        train_log_from_stdout(args.from_stdout, args.out_dirs[0])
    for d in args.out_dirs:
        for path in plot_run(d, args.plot_dir):
            print(f"saved {path}")
    if len(args.out_dirs) > 1:
        print(f"saved {plot_compare(args.out_dirs, args.plot_dir)}")
