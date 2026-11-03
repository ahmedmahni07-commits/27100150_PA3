"""Task 3 Step 5: adaptive sampling as an offline stopping simulation over the saved rollout orders.

    python -m task3_ttc.adaptive --config configs/task3.yaml

At n in {4, 8, 16} the controller looks only at the revealed prefix; otherwise it stops at 32. Every policy
returns the current self-consistency answer.
  agreement(tau):            unique non-NO_ANSWER SC leader with votes / n >= tau
  quality(tau, gamma):       agreement AND SC winner == RWC winner AND RWC margin >= gamma
                             (RWC score: self-certainty K for MATH, VisConf Z for MathVista; scores, VisConf
                             calibration and ranks recomputed on revealed candidates only)
Thresholds are chosen on the dev split (lowest mean token cost with dev accuracy >= fixed-32 SC - 1 pp; ties:
higher accuracy, then smaller tau, then smaller gamma); the always-32 policy is part of every grid as fallback.
Frozen settings are then evaluated once on the eval split.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_cfg, load_json, save_json
from task3_ttc.offline import load_questions
from task3_ttc.selection import NO_ANSWER, is_correct, rwc_margin, rwc_weights, visconf_scores, vote_winner


def checkpoint_stats(cfg, key) -> pd.DataFrame:
    vlm = cfg["models"][key]["type"] == "vlm"
    qs = load_questions(cfg, key)
    with open(Path(cfg["paths"]["outputs"]) / key / "orders.pkl", "rb") as f:
        orders = pickle.load(f)
    pts = cfg["adaptive"]["check_points"] + [cfg["adaptive"]["max_n"]]
    rows = []
    for qid, q in qs.items():
        full_correct_frac = float(q.correct.mean())
        for o, perm in enumerate(orders[qid]):
            for n in pts:
                cand = perm[:n]
                g = q.groups[cand]
                sc, tot = vote_winner(g, np.ones(n))
                top = max(tot.values())
                unique = sum(1 for v in tot.values() if v == top) == 1
                score = visconf_scores(q, cand)[0] if vlm else q.K[cand]
                rw, rtot = vote_winner(g, rwc_weights(score))
                rows.append({"qid": qid, "order": o, "n": n, "sc": sc, "sc_frac": top / n, "unique": unique,
                             "rwc": rw, "margin": rwc_margin(rtot), "correct": is_correct(q, sc),
                             "cost": int(q.n_tokens[cand].sum()), "pool_correct_frac": full_correct_frac})
    return pd.DataFrame(rows)


def simulate(st: pd.DataFrame, kind: str, tau: float | None, gamma: float | None, max_n: int) -> pd.DataFrame:
    """Return one row per (qid, order): stop n, cost, correctness."""
    if kind == "fixed32":
        stop = st["n"] == max_n
    else:
        ok = (st["unique"]) & (st["sc"] != NO_ANSWER) & (st["sc_frac"] >= tau - 1e-12)
        if kind == "quality":
            ok &= (st["rwc"] == st["sc"]) & (st["margin"] >= gamma - 1e-12)
        stop = ok | (st["n"] == max_n)
    s = st[stop].sort_values("n").groupby(["qid", "order"], as_index=False).first()
    final = st[st["n"] == max_n].set_index(["qid", "order"])["correct"]
    s["fixed32_correct"] = final.loc[list(zip(s["qid"], s["order"]))].to_numpy()
    return s


def policy_metrics(sim: pd.DataFrame) -> dict:
    perq = sim.groupby("qid")[["correct", "cost", "n"]].mean()
    return {"acc": 100 * perq["correct"].mean(), "cost": perq["cost"].mean(), "rollouts": perq["n"].mean()}


def grid(cfg):
    a = cfg["adaptive"]
    out = [("fixed32", None, None)]
    out += [("agreement", t, None) for t in a["tau"]]
    out += [("quality", t, g) for t, g in itertools.product(a["tau"], a["gamma"])]
    return out


def choose(rows: list[dict], ref_acc: float, tol: float, family: str) -> dict:
    cands = [r for r in rows if r["kind"] in (family, "fixed32") and r["acc"] >= ref_acc - tol - 1e-9]
    cands.sort(key=lambda r: (r["cost"], -r["acc"], r["tau"] if r["tau"] is not None else 9,
                              r["gamma"] if r["gamma"] is not None else 9))
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    cfg = load_cfg(ap.parse_args().config)
    splits = load_json(Path(cfg["paths"]["results"]) / "splits.json")
    max_n, tol = cfg["adaptive"]["max_n"], cfg["adaptive"]["tolerance_pp"]
    for key in cfg["models"]:
        out = ensure_dir(Path(cfg["paths"]["results"]) / "adaptive" / key)
        st = checkpoint_stats(cfg, key)
        st.to_parquet(Path(cfg["paths"]["outputs"]) / key / "adaptive_checkpoints.parquet")
        dev = st[st["qid"].isin(set(splits[key]["dev"]))]
        ev = st[st["qid"].isin(set(splits[key]["eval"]))]
        rows = []
        for kind, tau, gamma in grid(cfg):
            m = policy_metrics(simulate(dev, kind, tau, gamma, max_n))
            rows.append({"kind": kind, "tau": tau, "gamma": gamma, **m})
        devgrid = pd.DataFrame(rows)
        devgrid.to_csv(out / "dev_grid.csv", index=False)
        ref = devgrid[devgrid["kind"] == "fixed32"]["acc"].iloc[0]
        chosen = {fam: choose(rows, ref, tol, fam) for fam in ["agreement", "quality"]}
        save_json({"dev_fixed32_acc": ref, "tolerance_pp": tol, "chosen": chosen}, out / "selected_settings.json")

        # held-out evaluation with frozen settings
        res, stops, early_wrong = [], [], []
        for name, (kind, tau, gamma) in {"fixed32": ("fixed32", None, None),
                                         **{f: (c["kind"], c["tau"], c["gamma"]) for f, c in chosen.items()}}.items():
            sim = simulate(ev, kind, tau, gamma, max_n)
            res.append({"policy": name, "kind": kind, "tau": tau, "gamma": gamma, **policy_metrics(sim)})
            dist = sim["n"].value_counts(normalize=True).sort_index()
            stops += [{"policy": name, "stop_n": int(n), "frac": float(f)} for n, f in dist.items()]
            bad = sim[(sim["n"] < max_n) & (~sim["correct"])]
            early_wrong.append({"policy": name, "early_stop_rate": float((sim["n"] < max_n).mean()),
                                "incorrect_early_stops": int(len(bad)),
                                "recoverable_by_fixed32": int(bad["fixed32_correct"].sum()),
                                "mean_pool_correct_frac_of_incorrect_early": float(bad["pool_correct_frac"].mean())
                                if len(bad) else None})
            sim.to_parquet(out / f"eval_sim_{name}.parquet")
        # fixed-N SC reference at matching costs, from the offline run
        po = Path(cfg["paths"]["results"]) / "offline" / key / "per_question.parquet"
        if po.exists():
            pq = pd.read_parquet(po)
            fx = pq[(pq["method"] == "sc") & pq["qid"].isin(set(splits[key]["eval"]))].groupby("N")[["acc", "cost"]].mean()
            fx["acc"] *= 100
            fx.reset_index().to_csv(out / "fixed_N_sc_eval.csv", index=False)
        pd.DataFrame(res).to_csv(out / "eval_results.csv", index=False)
        pd.DataFrame(stops).to_csv(out / "stop_distribution.csv", index=False)
        pd.DataFrame(early_wrong).to_csv(out / "early_stop_errors.csv", index=False)
        print(f"\n== {key}\n", pd.DataFrame(res).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
