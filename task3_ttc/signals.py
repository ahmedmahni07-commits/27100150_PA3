"""Task 3 Step 3: what the quality signals capture (N = 32, held-out eval split by default).

    python -m task3_ttc.signals --config configs/task3.yaml [--split eval]

Writes results/task3/signals/<key>/:
  associations.csv     mean within-question Spearman(length, score) and Spearman(score, correct)
  quartiles.csv        within-question top-quartile minus bottom-quartile correctness, averaged over questions
  conf_vision.csv      (MathVista) per-rollout corr_t(c_t, v_t) summarised across questions; check Z_tok - Z = Cov
  trajectories/*.png   fixed illustrative trajectories (rule: first 2 split questions with >=1 correct and >=1
                       wrong rollout; per question the highest- and lowest-Z rollout)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.utils import ensure_dir, load_cfg, load_json, save_json
from task3_ttc.offline import load_questions
from task3_ttc.selection import visconf_scores


def _spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).statistic


def analyze(cfg, key, split):
    vlm = cfg["models"][key]["type"] == "vlm"
    qs = load_questions(cfg, key)
    ids = load_json(Path(cfg["paths"]["results"]) / "splits.json")[key][split]
    out = ensure_dir(Path(cfg["paths"]["results"]) / "signals" / key)
    assoc, quart, cv_rows, cov_err = [], [], [], 0.0
    for qid in ids:
        q = qs[qid]
        cand = np.arange(len(q.groups))
        S = {"L": q.L, "K": q.K, "neg_length": -q.n_tokens.astype(float)}
        if vlm:
            S["V"] = q.V
            S["Z"], S["Ztok"] = visconf_scores(q, cand)
        corr = q.correct.astype(float)
        length = q.n_tokens.astype(float)
        for name, s in S.items():
            assoc.append({"qid": qid, "score": name, "rho_len": _spearman(length, s), "rho_correct": _spearman(s, corr)})
            order = np.argsort(-s, kind="stable")
            k = len(s) // 4
            quart.append({"qid": qid, "score": name, "top_q": corr[order[:k]].mean(), "bottom_q": corr[order[-k:]].mean()})
        if vlm:
            pool = np.concatenate(q.kl_tok)
            q05, q95 = np.percentile(pool, [5, 95])
            for i in cand:
                c = np.clip((q.kl_tok[i] - q05) / max(q95 - q05, 1e-8), 0, 1)
                v = q.v_tok[i]
                if len(c) < 3:
                    continue
                r = np.corrcoef(c, v)[0, 1] if np.std(c) > 0 and np.std(v) > 0 else np.nan
                cov = np.mean(c * v) - np.mean(c) * np.mean(v)
                cov_err = max(cov_err, abs((S["Ztok"][i] - S["Z"][i]) - cov) if np.isclose(q.V[i], v.mean()) else 0.0)
                cv_rows.append({"qid": qid, "rollout": int(i), "corr_c_v": r, "cov_c_v": cov,
                                "correct": bool(q.correct[i])})
    a = pd.DataFrame(assoc).groupby("score")[["rho_len", "rho_correct"]].mean().reset_index()
    qd = pd.DataFrame(quart)
    qd["diff"] = qd["top_q"] - qd["bottom_q"]
    qs_ = qd.groupby("score")[["top_q", "bottom_q", "diff"]].mean().mul(100).reset_index()
    a.to_csv(out / "associations.csv", index=False)
    qs_.to_csv(out / "quartiles.csv", index=False)
    print(key, "\n", a.round(3).to_string(index=False), "\n", qs_.round(2).to_string(index=False))
    if vlm:
        cv = pd.DataFrame(cv_rows)
        cv.to_csv(out / "conf_vision_per_rollout.csv", index=False)
        per_q = cv.groupby("qid")[["corr_c_v", "cov_c_v"]].mean()
        summ = {"mean_within_rollout_corr": float(cv["corr_c_v"].mean()),
                "median_within_rollout_corr": float(cv["corr_c_v"].median()),
                "frac_rollouts_positive_corr": float((cv["corr_c_v"] > 0).mean()),
                "mean_of_question_means_corr": float(per_q["corr_c_v"].mean()),
                "corr_by_correct": cv.groupby("correct")["corr_c_v"].mean().to_dict(),
                "max_abs_err_Ztok_minus_Z_vs_cov": cov_err}
        save_json(summ, out / "conf_vision_summary.json")
        print(summ)
        plot_trajectories(qs, ids, out)


def plot_trajectories(qs, ids, out):
    td = ensure_dir(out / "trajectories")
    picked = [qid for qid in ids if qs[qid].correct.any() and not qs[qid].correct.all()][:2]
    for qid in picked:
        q = qs[qid]
        cand = np.arange(len(q.groups))
        Z, _ = visconf_scores(q, cand)
        pool = np.concatenate(q.kl_tok)
        q05, q95 = np.percentile(pool, [5, 95])
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=False)
        for ax, i, tag in [(axes[0], int(np.argmax(Z)), "highest Z"), (axes[1], int(np.argmin(Z)), "lowest Z")]:
            c = np.clip((q.kl_tok[i] - q05) / max(q95 - q05, 1e-8), 0, 1)
            w = max(1, len(c) // 50)
            sm = lambda x: np.convolve(x, np.ones(w) / w, mode="valid")
            ax.plot(sm(c), label="calibrated confidence c_t")
            ax.plot(sm(q.v_tok[i] / max(q.v_tok[i].max(), 1e-12)), label="visual engagement v_t (scaled)")
            ax.set_title(f"{qid} rollout {i} ({tag}, {'correct' if q.correct[i] else 'wrong'}), {len(c)} tokens",
                         fontsize=9)
            ax.set_xlabel(f"generated token (moving average, window {w})")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(td / f"{qid}.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    for key in cfg["models"]:
        analyze(cfg, key, args.split)


if __name__ == "__main__":
    main()
