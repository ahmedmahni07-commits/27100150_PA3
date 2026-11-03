"""Task 1 tables and figures from saved results (CPU only).

    python -m task1_ft.summarize --config configs/task1.yaml

Writes results/task1/summary/{benchmarks.csv, resources.csv, regularization.csv} and figures/{losses.png, ...}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.utils import ensure_dir, load_cfg, load_json, read_jsonl

CONDS = ["baseline", "full_ft_625", "full_ft_1250", "l2sp_1250", "lora4_1250", "lora16_1250", "oplora16_1250"]
LABELS = {"baseline": "Baseline", "full_ft_625": "Full FT (mid)", "full_ft_1250": "Full FT", "l2sp_1250": "Full FT + L2-SP",
          "lora4_1250": "LoRA-4", "lora16_1250": "LoRA-16", "oplora16_1250": "OPLoRA-16"}
METHODS = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
BENCH = ["gsm8k", "hellaswag", "boolq", "arc_easy"]


def benchmarks(res: Path) -> pd.DataFrame:
    rows = []
    for c in CONDS:
        f = res / "eval" / c / "scores.json"
        if not f.exists():
            continue
        s = load_json(f)
        row = {"condition": LABELS[c]}
        for b in BENCH:
            if b in s:
                row[b] = 100 * s[b]["primary"]
                if b in ("hellaswag", "arc_easy"):
                    row[f"{b}_raw_acc"] = 100 * s[b]["acc"]
                if b == "gsm8k":
                    row["gsm8k_trunc_%"] = 100 * s[b]["truncation_rate"]
                    row["gsm8k_unextractable_%"] = 100 * s[b]["unextractable_rate"]
                    row["gsm8k_mean_tokens"] = s[b]["mean_gen_tokens"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and (df["condition"] == "Baseline").any():
        base = df[df["condition"] == "Baseline"].iloc[0]
        for b in BENCH:
            if b in df:
                df[f"{b}_delta_pp"] = df[b] - base[b]
    return df


def losses(res: Path, fig_dir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for m in METHODS:
        f = res / "train" / m / "log.jsonl"
        if not f.exists():
            continue
        d = pd.DataFrame(read_jsonl(f))
        sm = d["sft_loss"].rolling(25, min_periods=1).mean()
        ax[0].plot(d["update"], sm, label=m)
        ax[1].plot(d["update"], d["lr"], label=m)
    ax[0].set(xlabel="optimizer update", ylabel="SFT loss (25-update mean)", title="Response-token CE")
    ax[1].set(xlabel="optimizer update", ylabel="learning rate", yscale="log", title="Schedule")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "losses.png", dpi=200)


def regularization(res: Path, fig_dir: Path) -> pd.DataFrame:
    rows = []
    for m in ["full_ft", "l2sp"]:
        s = res / "train" / m / "summary.json"
        if s.exists():
            j = load_json(s)
            rows.append({"method": m, "displacement_625": j.get("displacement_625"),
                         "displacement_1250": j.get("displacement_1250")})
    f = res / "train" / "l2sp" / "log.jsonl"
    if f.exists():
        d = pd.DataFrame(read_jsonl(f))
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(d["update"], d["sft_loss"].rolling(25, min_periods=1).mean(), label="SFT loss")
        ax.plot(d["update"], d["lam_omega"], label="λΩ")
        ax.plot(d["update"], d["total_loss"].rolling(25, min_periods=1).mean(), label="total")
        ax.set(xlabel="optimizer update", ylabel="loss", title="L2-SP objective terms")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "l2sp_terms.png", dpi=200)
        if rows:
            rows[-1].update({"final_omega": d["omega"].iloc[-1], "final_lam_omega": d["lam_omega"].iloc[-1],
                             "final_lam_omega_over_sft": d["lam_omega"].iloc[-1] / d["sft_loss"].iloc[-25:].mean()})
    return pd.DataFrame(rows)


def resources(res: Path) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        s, i = res / "train" / m / "summary.json", res / "train" / m / "model_info.json"
        if not s.exists():
            continue
        j, info = load_json(s), load_json(i)
        sizes = j.get("checkpoint_sizes_mb", {})
        rows.append({"method": m, "trainable_params": info["trainable_params"],
                     "train_hours": j["train_seconds"] / 3600, "peak_gpu_gb": j["peak_gpu_mem_gb"],
                     "final_weights_mb": sizes.get("1250", {}).get("weights_mb"),
                     "resume_state_mb": sizes.get("1250", {}).get("resume_state_mb"),
                     "skipped_updates": j.get("skipped_updates"),
                     "oplora_svd_s": info.get("svd_preprocess_seconds"), "oplora_basis_mb": info.get("basis_storage_mb"),
                     "l2sp_ref_mb": info.get("l2sp_ref_storage_mb"), "l2sp_penalty_s": j.get("l2sp_penalty_seconds")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    cfg = load_cfg(ap.parse_args().config)
    res = Path(cfg["paths"]["results"])
    out, fig_dir = ensure_dir(res / "summary"), ensure_dir(res / "figures")
    for name, df in [("benchmarks", benchmarks(res)), ("resources", resources(res)),
                     ("regularization", regularization(res, fig_dir))]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name}\n", df.round(3).to_string(index=False))
    losses(res, fig_dir)


if __name__ == "__main__":
    main()
