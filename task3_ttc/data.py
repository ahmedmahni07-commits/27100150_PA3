"""Task 3 data: load/join the supplied rollout tables, verify token alignment, fix dev/eval splits.

    python -m task3_ttc.data inspect          # print schemas of every parquet file (fix configs/task3.yaml)
    python -m task3_ttc.data build            # join, verify alignment, save per-rollout table + splits

Outputs (outputs/task3/<key>/): rollouts.parquet (one row per rollout, scalar summaries) and tokens.pkl
(per-rollout token arrays kl / v used by VisConf); results/task3/splits.json and alignment.json.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_cfg, save_json


def pick(cols, aliases, required=True):
    for a in aliases:
        if a in cols:
            return a
    if required:
        raise KeyError(f"none of {aliases} in columns {list(cols)} – update configs/task3.yaml:columns")
    return None


def model_dir(cfg, key) -> Path:
    return Path(cfg["data_root"]) / cfg["models"][key]["folder"]


def inspect(cfg):
    for key in cfg["models"]:
        for f in sorted(model_dir(cfg, key).glob("*.parquet")):
            df = pd.read_parquet(f).head(2)
            print(f"\n== {key}/{f.name}: {len(pd.read_parquet(f, columns=[df.columns[0]]))} rows")
            for c in df.columns:
                v = df[c].iloc[0]
                desc = f"len={len(v)} first={list(v[:3])}" if hasattr(v, "__len__") and not isinstance(v, str) else repr(v)[:80]
                print(f"  {c:28s} {str(df[c].dtype):10s} {desc}")


def _arr(x):
    return np.asarray(x, dtype=np.float64) if x is not None else np.zeros(0)


def build(cfg, key) -> dict:
    C = cfg["columns"]
    d = model_dir(cfg, key)
    gen = pd.read_parquet(d / "generations.parquet")
    prob = pd.read_parquet(d / "probability_metrics.parquet")
    keys = [pick(gen.columns, C["dataset"], required=False), pick(gen.columns, C["question_id"]),
            pick(gen.columns, C["rollout_id"])]
    keys = [k for k in keys if k]
    df = gen.merge(prob, on=keys, how="inner", suffixes=("", "_prob"), validate="one_to_one")
    assert len(df) == len(gen) == len(prob), (len(df), len(gen), len(prob))
    is_vlm = cfg["models"][key]["type"] == "vlm"
    if is_vlm:
        att = pd.read_parquet(d / "attention_metrics.parquet")
        df = df.merge(att, on=keys, how="inner", suffixes=("", "_att"), validate="one_to_one")
        assert len(df) == len(gen)

    qcol, rcol = pick(df.columns, C["question_id"]), pick(df.columns, C["rollout_id"])
    lp_c, kl_c = pick(df.columns, C["logp"]), pick(df.columns, C["kl"])
    tok_c = pick(df.columns, C["token_ids"], required=False)
    nt_c = pick(df.columns, C["n_tokens"], required=False)
    ex_c, gold_c = pick(df.columns, C["extracted"]), pick(df.columns, C["gold"])
    txt_c = pick(df.columns, C["text"], required=False)
    prob_c = pick(df.columns, C["problem"], required=False)

    rows, tokens, align = [], {}, {"n_rollouts": len(df), "logp_kl_len_mismatch": 0, "logp_vs_token_ids_diff": {},
                                   "attn_len_mismatch": 0, "nonfinite_values": 0}
    for r in df.itertuples(index=False):
        r = r._asdict()
        lp, kl = _arr(r[lp_c]), _arr(r[kl_c])
        if len(lp) != len(kl):
            align["logp_kl_len_mismatch"] += 1
        if tok_c:
            diff = len(r[tok_c]) - len(lp)
            align["logp_vs_token_ids_diff"][str(diff)] = align["logp_vs_token_ids_diff"].get(str(diff), 0) + 1
        align["nonfinite_values"] += int((~np.isfinite(lp)).sum() + (~np.isfinite(kl)).sum())
        v = None
        if is_vlm:
            ai, ap, ag = (_arr(r[pick(df.columns, C[c])]) for c in ("attn_image", "attn_prompt", "attn_generated"))
            if not (len(ai) == len(ap) == len(ag) == len(lp)):
                align["attn_len_mismatch"] += 1
            v = ai / np.maximum(ai + ap + ag, 1e-12)
        T = int(r[nt_c]) if nt_c else len(lp)
        qid, rid = str(r[qcol]), int(r[rcol])
        ex = r[ex_c]
        ex = None if ex is None or (isinstance(ex, float) and np.isnan(ex)) or str(ex).strip() == "" else str(ex)
        rows.append({"qid": qid, "rollout_id": rid, "n_tokens": T, "n_scored_tokens": len(lp),
                     "L": float(np.mean(lp)) if len(lp) else -np.inf,
                     "K": float(np.mean(kl)) if len(kl) else -np.inf,
                     "V": float(np.mean(v)) if v is not None and len(v) else np.nan,
                     "extracted": ex, "gold": str(r[gold_c]),
                     "text": r[txt_c] if txt_c else None, "problem": r[prob_c] if prob_c else None})
        tokens[(qid, rid)] = {"kl": kl.astype(np.float32), "v": None if v is None else v.astype(np.float32)}
    out = pd.DataFrame(rows).sort_values(["qid", "rollout_id"]).reset_index(drop=True)
    counts = out.groupby("qid").size()
    align["rollouts_per_question"] = counts.value_counts().to_dict()
    align["n_questions"] = int(out["qid"].nunique())
    od = ensure_dir(Path(cfg["paths"]["outputs"]) / key)
    out.to_parquet(od / "rollouts.parquet")
    with open(od / "tokens.pkl", "wb") as f:
        pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    return align


def make_splits(qids: list[str], cfg) -> dict:
    """Sort IDs lexicographically, permute with default_rng(6304): first 200 dev, remaining 800 eval."""
    s = sorted(qids)
    perm = np.random.default_rng(cfg["seed"]).permutation(len(s))
    ordered = [s[i] for i in perm]
    n_dev = cfg["splits"]["n_dev"]
    return {"dev": ordered[:n_dev], "eval": ordered[n_dev:], "live_math": ordered[n_dev:][: cfg["splits"]["n_live"]]}


def load_pool(cfg, key):
    od = Path(cfg["paths"]["outputs"]) / key
    df = pd.read_parquet(od / "rollouts.parquet")
    with open(od / "tokens.pkl", "rb") as f:
        tokens = pickle.load(f)
    return df, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("step", choices=["inspect", "build"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.step == "inspect":
        inspect(cfg)
        return
    res = ensure_dir(cfg["paths"]["results"])
    splits, align = {}, {}
    for key in cfg["models"]:
        align[key] = build(cfg, key)
        df = pd.read_parquet(Path(cfg["paths"]["outputs"]) / key / "rollouts.parquet", columns=["qid"])
        sp = make_splits(df["qid"].unique().tolist(), cfg)
        if key != "math":
            sp.pop("live_math")
        splits[key] = sp
        print(key, align[key])
    save_json(align, res / "alignment.json")
    save_json(splits, res / "splits.json")


if __name__ == "__main__":
    main()
