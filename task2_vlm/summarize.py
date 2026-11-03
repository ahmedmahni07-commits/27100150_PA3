"""Task 2 learning-curve figure and resource table (CPU).

    python -m task2_vlm.summarize --config configs/task2.yaml [--n-vis 49]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.utils import ensure_dir, load_cfg, load_json, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--n-vis", type=int, default=196)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    res = Path(cfg["paths"]["results"]) / f"vis{args.n_vis}"
    figs = ensure_dir(res / "figures")
    tr, dv = pd.DataFrame(read_jsonl(res / "train_log.jsonl")), pd.DataFrame(read_jsonl(res / "dev_log.jsonl"))
    bounds = pd.Series([s["updates"] for s in cfg["train"]["stages"]]).cumsum().tolist()[:-1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
    ax[0].plot(tr["update"], tr["loss"].rolling(25, min_periods=1).mean())
    ax[0].set(title="training loss (25-update mean)", xlabel="update")
    ax[1].plot(dv["update"], dv["dev_caption_loss"], "o-")
    ax[1].set(title="dev caption loss", xlabel="update")
    for suite, ls in [("grounding_dev", "-"), ("vqa_dev", "--")]:
        for cond in ["correct", "swap", "none"]:
            ax[2].plot(dv["update"], dv[f"{suite}_nll_{cond}"], ls, marker="o", ms=3, label=f"{suite.split('_')[0]} {cond}")
        ax[3].plot(dv["update"], dv[f"{suite}_swap_penalty"], ls, marker="o", label=suite.split("_")[0])
    ax[2].set(title="reference-answer NLL", xlabel="update")
    ax[3].axhline(0, color="grey", lw=0.8)
    ax[3].set(title="image-swap penalty ΔNLL", xlabel="update")
    ax[2].legend(fontsize=6)
    ax[3].legend(fontsize=7)
    for a in ax:
        for b in bounds:
            a.axvline(b, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(figs / "learning_curves.png", dpi=200)
    rows = []
    timing = load_json(res / "timing.json")
    for st in cfg["train"]["stages"]:
        p = res / f"params_{st['name']}.json"
        if p.exists():
            j = load_json(p)
            rows.append({"stage": st["name"], "updates": st["updates"], **j,
                         "hours": timing.get(st["name"], {}).get("seconds", float("nan")) / 3600,
                         "peak_gpu_gb": timing.get(st["name"], {}).get("peak_gpu_mem_gb")})
    fc = Path(cfg["paths"]["results"]) / "feature_caching.json"
    df = pd.DataFrame(rows)
    df.to_csv(res / "resources.csv", index=False)
    print(df.to_string(index=False))
    if fc.exists():
        print("feature caching:", load_json(fc))


if __name__ == "__main__":
    main()
