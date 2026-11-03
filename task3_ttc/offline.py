"""Task 3 Steps 1–2: repeated-sampling opportunity and selection rules on the supplied rollouts.

    python -m task3_ttc.offline --config configs/task3.yaml

For every question, order (10 seeded permutations) and budget N in {1,2,4,8,16,32}, every method selects an
answer from the first N rollouts of that order. Outcomes are averaged over orders within a question, then over
questions. Outputs (results/task3/offline/<key>/):
  per_question.parquet   question x N x method: accuracy, oracle, cost, distinct groups (order-averaged)
  per_order.parquet      raw per-order rows (selected rollout/group, correctness, cost)
  summary.csv            split x N x method means; failure decomposition; bootstrap CIs vs SC / FirstAns
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_cfg, load_json
from task3_ttc.selection import NO_ANSWER, Question, is_correct, make_orders, method_names, select_all


def load_questions(cfg, key) -> dict[str, Question]:
    od = Path(cfg["paths"]["outputs"]) / key
    df = pd.read_parquet(od / "rollouts.parquet", columns=["qid", "rollout_id", "n_tokens", "L", "K", "V"])
    gr = pd.read_parquet(od / "groups.parquet")
    df = df.merge(gr, on=["qid", "rollout_id"], validate="one_to_one").sort_values(["qid", "rollout_id"])
    with open(od / "tokens.pkl", "rb") as f:
        tok = pickle.load(f)
    vlm = cfg["models"][key]["type"] == "vlm"
    qs = {}
    for qid, g in df.groupby("qid", sort=True):
        rids = g["rollout_id"].tolist()
        qs[qid] = Question(qid=qid, groups=g["group"].to_numpy(int), correct=g["correct"].to_numpy(bool),
                           n_tokens=g["n_tokens"].to_numpy(int), L=g["L"].to_numpy(float), K=g["K"].to_numpy(float),
                           V=g["V"].to_numpy(float) if vlm else None,
                           kl_tok=[tok[(qid, r)]["kl"] for r in rids],
                           v_tok=[tok[(qid, r)]["v"] for r in rids] if vlm else None)
    return qs


def run(cfg, key):
    vlm = cfg["models"][key]["type"] == "vlm"
    qs = load_questions(cfg, key)
    R = len(next(iter(qs.values())).groups)
    orders = make_orders(list(qs), R, cfg["n_orders"], cfg["seed"])
    od = ensure_dir(Path(cfg["paths"]["outputs"]) / key)
    with open(od / "orders.pkl", "wb") as f:
        pickle.dump(orders, f)
    methods = method_names(vlm)
    rows = []
    for qid, q in qs.items():
        for o, perm in enumerate(orders[qid]):
            for N in cfg["budgets"]:
                cand = perm[:N]
                oracle = bool(q.correct[cand].any())
                n_groups = len(set(q.groups[cand].tolist()))
                sel = select_all(q, cand, vlm)
                for m in methods:
                    g, cost, extra = sel[m]
                    sel_pos = int(np.flatnonzero(q.groups[cand] == g)[0])
                    rows.append({"qid": qid, "order": o, "N": N, "method": m, "group": g,
                                 "selected_rollout": int(cand[sel_pos]), "correct": is_correct(q, g),
                                 "no_answer": g == NO_ANSWER, "cost": cost, "oracle": oracle, "n_groups": n_groups})
    per_order = pd.DataFrame(rows)
    return per_order


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Paired question-level bootstrap of mean(a - b); returns (diff, lo, hi) in percentage points."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def summarize(per_order: pd.DataFrame, splits: dict, cfg, out: Path):
    pq = per_order.groupby(["qid", "N", "method"], as_index=False).agg(
        acc=("correct", "mean"), oracle=("oracle", "mean"), cost=("cost", "mean"), n_groups=("n_groups", "mean"),
        no_answer=("no_answer", "mean"))
    pq.to_parquet(out / "per_question.parquet")
    rows, ci_rows = [], []
    for split in ["dev", "eval"]:
        sub = pq[pq["qid"].isin(set(splits[split]))]
        agg = sub.groupby(["N", "method"], as_index=False).mean(numeric_only=True)
        agg["split"] = split
        # failure decomposition: wrong with no correct candidate vs wrong despite a correct candidate
        po = per_order[per_order["qid"].isin(set(splits[split]))].copy()
        po["sampling_fail"] = ~po["oracle"]
        po["selection_fail"] = po["oracle"] & ~po["correct"]
        dec = (po.groupby(["qid", "N", "method"])[["sampling_fail", "selection_fail"]].mean()
               .groupby(["N", "method"]).mean().reset_index())
        rows.append(agg.merge(dec, on=["N", "method"]))
        for N in cfg["budgets"]:
            piv = sub[sub["N"] == N].pivot(index="qid", columns="method", values="acc").sort_index()
            for ref in ["sc", "firstans"]:
                for m in piv.columns:
                    if m == ref:
                        continue
                    d, lo, hi = bootstrap_diff(piv[m].to_numpy(), piv[ref].to_numpy(), cfg["n_bootstrap"], cfg["seed"])
                    ci_rows.append({"split": split, "N": N, "method": m, "vs": ref, "diff_pp": d, "ci_lo": lo, "ci_hi": hi})
    summ = pd.concat(rows)
    for c in ["acc", "oracle", "no_answer", "sampling_fail", "selection_fail"]:
        summ[c] = 100 * summ[c]
    summ.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(out / "bootstrap_vs_ref.csv", index=False)
    # bootstrap CI for each method's own accuracy
    own = []
    for split in ["dev", "eval"]:
        sub = pq[pq["qid"].isin(set(splits[split]))]
        for (N, m), g in sub.groupby(["N", "method"]):
            a = g.sort_values("qid")["acc"].to_numpy()
            d, lo, hi = bootstrap_diff(a, np.zeros_like(a), cfg["n_bootstrap"], cfg["seed"])
            own.append({"split": split, "N": N, "method": m, "acc": d, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(own).to_csv(out / "accuracy_ci.csv", index=False)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--only", default=None, help="math | mathvista")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    splits = load_json(Path(cfg["paths"]["results"]) / "splits.json")
    for key in cfg["models"]:
        if args.only and key != args.only:
            continue
        out = ensure_dir(Path(cfg["paths"]["results"]) / "offline" / key)
        po = run(cfg, key)
        po.to_parquet(Path(cfg["paths"]["outputs"]) / key / "per_order.parquet")
        summ = summarize(po, splits[key], cfg, out)
        print(f"\n== {key} (eval split)\n",
              summ[summ.split == "eval"].pivot(index="method", columns="N", values="acc").round(1).to_string())


if __name__ == "__main__":
    main()
