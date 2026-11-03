"""Task 3 figures: accuracy / oracle coverage vs N and vs token cost; adaptive policies vs fixed budgets.

    python -m task3_ttc.plots --config configs/task3.yaml [--split eval]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.utils import ensure_dir, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    res = Path(cfg["paths"]["results"])
    figs = ensure_dir(res / "figures")
    keys = list(cfg["models"])
    fig, axes = plt.subplots(2, len(keys), figsize=(6 * len(keys), 7))
    for j, key in enumerate(keys):
        s = pd.read_csv(res / "offline" / key / "summary.csv")
        s = s[s["split"] == args.split]
        for ax, x in [(axes[0, j], "N"), (axes[1, j], "cost")]:
            orc = s[s["method"] == "sc"]
            ax.plot(orc[x], orc["oracle"], "k--", label="oracle coverage")
            for m in sorted(s["method"].unique()):
                d = s[s["method"] == m].sort_values("N")
                ax.plot(d[x], d["acc"], marker="o", ms=3, label=m)
            ax.set_xscale("log", base=2)
            ax.set(xlabel="candidates N" if x == "N" else "mean generated tokens / question", ylabel="accuracy (%)",
                   title=f"{key} ({args.split})")
        ad = res / "adaptive" / key / "eval_results.csv"
        if ad.exists():
            a = pd.read_csv(ad)
            axes[1, j].scatter(a["cost"], a["acc"], marker="*", s=120, c="red", zorder=5)
            for _, r in a.iterrows():
                axes[1, j].annotate(r["policy"], (r["cost"], r["acc"]), fontsize=7)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "accuracy_vs_compute.png", dpi=200)
    print("saved", figs / "accuracy_vs_compute.png")


if __name__ == "__main__":
    main()
