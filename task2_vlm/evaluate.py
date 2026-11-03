"""Task 2 final evaluation on both final-test suites under four image conditions.

    python -m task2_vlm.evaluate --config configs/task2.yaml [--n-vis 49] [--stage 3]

Conditions: correct | swap (partner image of the complementary pair) | mismatched (fixed unrelated image from
another pair of the same suite) | none (visual tokens removed). Question text and references are fixed.
Greedy decoding, thinking disabled, 16 new tokens, batch 8.
Scoring: grounding -> normalised exact match (presence / counting separately); VQAv2 -> local soft VQA accuracy
against all ten human answers (yes/no, number, other). Pair accuracy, identical-answer rate, empty-answer rate,
follow-partner rate, mean reference NLL and generation time per answer are reported for every suite x condition.
Correct-vs-swap advantages get 95% CIs from 4,000 bootstrap resamples of complete pairs (same pair IDs across
conditions).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.utils import ensure_dir, load_cfg, save_json, set_seed
from task1_ft.methods import inject_adapters, load_adapter_state
from task2_vlm.common import (Features, build_vlm, flatten_pairs, load_sets, load_tokenizer, mismatch_map, qa_prompt,
                              teacher_forced_nll)
from task2_vlm.model import PromptBuilder, collate
from task2_vlm.vqa_norm import normalize_answer, vqa_soft_accuracy

SUITES = {"grounding": "grounding_test", "vqa": "vqa_test"}
CONDS = ["correct", "swap", "mismatched", "none"]


def load_trained(cfg, n_vis, stage, device):
    vlm = build_vlm(cfg, device)
    out = Path(cfg["paths"]["outputs"]) / f"vis{n_vis}"
    f = sorted(out.glob(f"stage{stage}_*.pt"))[0]
    sd = torch.load(f, map_location=device)
    if sd["adapters"]:
        lc = cfg["model"]["lora"]
        inject_adapters(vlm.llm, "lora", lc["r"], lc["alpha"], 0.0, lc["targets"])
        vlm.llm.to(device)
        load_adapter_state(vlm.llm, sd["adapters"])
    vlm.connector.load_state_dict(sd["connector"])
    return vlm.eval(), str(f)


def image_for(r, cond, mm):
    return {"correct": r["image_id"], "swap": r["partner_image_id"],
            "mismatched": mm.get(f"{r['pair_id']}|{r['slot']}"), "none": None}[cond]


@torch.no_grad()
def generate(vlm, pb, feats, rows, cond, mm, cfg, device):
    bs, preds, secs = cfg["eval"]["batch_size"], [], []
    n_vis = 0 if cond == "none" else feats.n_vis
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        ex = [pb.build(qa_prompt(cfg, r["question"]), None, n_vis, 0) for r in chunk]
        ids, _, att, vis = (t.to(device) for t in collate(ex, pb.tok.pad_token_id, left_pad=True))
        f = None if cond == "none" else feats.get([image_for(r, cond, mm) for r in chunk]).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            gen = vlm.greedy(ids, att, vis, f, cfg["eval"]["max_new_tokens"], pb.eot, pb.tok.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / len(chunk)
        for g in gen.tolist():
            preds.append(pb.tok.decode(g, skip_special_tokens=True).strip())
            secs.append(dt)
    return preds, secs


def score(row, pred, suite):
    if suite == "vqa":
        return vqa_soft_accuracy(pred, row["answers"])
    return float(normalize_answer(pred) == normalize_answer(row["answer"]))


def pair_boot(df: pd.DataFrame, col: str, n: int, seed: int) -> tuple[float, float, float]:
    """Correct-minus-swap advantage, bootstrapping complete pairs (both questions of a pair together)."""
    piv = df.pivot_table(index=["pair_id", "slot"], columns="condition", values=col)
    d = (piv["correct"] - piv["swap"]).groupby(level=0).mean() if col != "nll" else \
        (piv["swap"] - piv["correct"]).groupby(level=0).mean()   # NLL: positive = correct image helps
    rng = np.random.default_rng(seed)
    v = d.to_numpy()
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--n-vis", type=int, default=None)
    ap.add_argument("--stage", type=int, default=3)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    n_vis = args.n_vis or cfg["model"]["n_visual_tokens"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    res = ensure_dir(Path(cfg["paths"]["results"]) / f"vis{n_vis}" / f"final_stage{args.stage}")
    tok = load_tokenizer(cfg["llm"])
    pb = PromptBuilder(tok, cfg["caps"], cfg.get("enable_thinking", False))
    sets, feats = load_sets(cfg), Features(cfg, n_vis)
    vlm, ckpt = load_trained(cfg, n_vis, args.stage, device)
    all_rows, summary, ci = [], [], []
    for suite, name in SUITES.items():
        rows = flatten_pairs(sets[name])
        mm = mismatch_map(rows, cfg["seed"])
        save_json(mm, res / f"mismatch_map_{suite}.json")
        for cond in CONDS:
            preds, secs = generate(vlm, pb, feats, rows, cond, mm, cfg, device)
            key = {"correct": "image_id", "swap": "partner_image_id", "none": None, "mismatched": "_mm"}[cond]
            for r in rows:
                r["_mm"] = mm[f"{r['pair_id']}|{r['slot']}"]
            nll = teacher_forced_nll(vlm, pb, feats, rows, key, cfg, device, lambda r: qa_prompt(cfg, r["question"]),
                                     cfg["caps"]["answer"], True)
            for r, p, s, l in zip(rows, preds, secs, nll):
                all_rows.append({"suite": suite, "condition": cond, "pair_id": r["pair_id"], "slot": r["slot"],
                                 "kind": r["kind"], "question": r["question"], "image_used": image_for(r, cond, mm),
                                 "reference": r["answer"], "partner_reference": r["partner_answer"], "pred": p,
                                 "score": score(r, p, suite),
                                 "match_canonical": float(normalize_answer(p) == normalize_answer(r["answer"])),
                                 "follows_partner": float(
                                     normalize_answer(p) == normalize_answer(r["partner_answer"])),
                                 "empty": float(p.strip() == ""), "nll": float(l), "gen_seconds": s})
    df = pd.DataFrame(all_rows)
    df.to_csv(res / "predictions.csv", index=False)
    for (suite, cond), g in df.groupby(["suite", "condition"]):
        pair = g.groupby("pair_id").agg(both=("match_canonical", lambda x: float((x == 1).all())),
                                        ident=("pred", lambda x: float(normalize_answer(x.iloc[0]) == normalize_answer(x.iloc[1]))))
        row = {"suite": suite, "condition": cond, "accuracy": 100 * g["score"].mean(),
               "pair_accuracy": 100 * pair["both"].mean(), "identical_answer_rate": 100 * pair["ident"].mean(),
               "empty_rate": 100 * g["empty"].mean(), "follows_partner_rate": 100 * g["follows_partner"].mean(),
               "mean_ref_nll": g["nll"].mean(), "gen_ms_per_answer": 1000 * g["gen_seconds"].mean(), "n": len(g)}
        for k, gk in g.groupby("kind"):
            row[f"acc_{k}"] = 100 * gk["score"].mean()
        summary.append(row)
    for suite, g in df.groupby("suite"):
        for col in ["score", "nll"]:
            d, lo, hi = pair_boot(g, col, cfg["eval"]["n_bootstrap"], cfg["seed"])
            scale = 100 if col == "score" else 1
            ci.append({"suite": suite, "metric": "accuracy_pp" if col == "score" else "nll_swap_minus_correct",
                       "advantage": scale * d, "ci_lo": scale * lo, "ci_hi": scale * hi})
        for kind, gk in g.groupby("kind"):
            d, lo, hi = pair_boot(gk, "score", cfg["eval"]["n_bootstrap"], cfg["seed"])
            ci.append({"suite": suite, "metric": f"accuracy_pp[{kind}]", "advantage": 100 * d, "ci_lo": 100 * lo,
                       "ci_hi": 100 * hi})
    pd.DataFrame(summary).to_csv(res / "summary.csv", index=False)
    pd.DataFrame(ci).to_csv(res / "correct_vs_swap_bootstrap.csv", index=False)
    save_json({"checkpoint": ckpt, "n_vis": n_vis}, res / "meta.json")
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    print(pd.DataFrame(ci).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
