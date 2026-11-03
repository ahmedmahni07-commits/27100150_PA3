"""Answer normalisation, equivalence groups and correctness labels (Task 3).

Groups are formed per question WITHOUT gold answers; the same equivalence rule then grades each group.
  * MATH: math-verify (`parse` + `verify`); answers are wrapped as \\boxed{...} before parsing.
  * MathVista: rules of MathVista's released scorer using testmini.json metadata – multiple choice ->
    option letter; integer -> int; float -> rounded to the question's precision; text/list -> stripped string.
Missing extractions form one NO_ANSWER group (group id -1) that is always graded incorrect but still votes.

    python -m task3_ttc.grading --config configs/task3.yaml
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from common.utils import ensure_dir, load_cfg, save_json

NO_ANSWER = -1


# ---------------------------------------------------------------- MATH
@lru_cache(maxsize=200_000)
def _parse_math(s: str):
    from math_verify import parse

    s = s.strip()
    return parse(s if "\\boxed" in s else f"\\boxed{{{s}}}", parsing_timeout=5)


@lru_cache(maxsize=500_000)
def math_equiv(a: str, b: str) -> bool:
    from math_verify import verify

    if a.strip() == b.strip():
        return True
    pa, pb = _parse_math(a), _parse_math(b)
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb, timeout_seconds=5))
    except Exception:
        return False


# ---------------------------------------------------------------- MathVista
def load_mathvista_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def mathvista_normalize(ans: str | None, meta: dict) -> str | None:
    """MathVista `normalize_extracted_answer`, except that multiple choice is kept as an option letter."""
    if ans is None:
        return None
    s = str(ans).strip()
    if s == "":
        return None
    qt, at = meta["question_type"], meta["answer_type"]
    if qt == "multi_choice":
        choices = meta.get("choices") or []
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.findall(r"\(([a-zA-Z])\)", s)
        if m:
            s = m[0].upper()
        if s.upper() in letters and len(s) == 1:
            return s.upper()
        low = [str(c).strip().lower() for c in choices]
        if s.lower() in low:
            return letters[low.index(s.lower())]
        if not choices:
            return s
        # MathVista falls back to the most similar option (edit-distance); difflib ratio is the same idea
        best = max(range(len(choices)), key=lambda i: difflib.SequenceMatcher(None, s.lower(), low[i]).ratio())
        return letters[best]
    try:
        if at == "integer":
            return str(int(float(s.replace(",", ""))))
        if at == "float":
            return str(round(float(s.replace(",", "")), int(meta.get("precision") or 0)))
    except (ValueError, OverflowError):
        return None  # unparsable number -> treated as missing extraction (NO_ANSWER)
    return s.lower() if at == "text" else s


def mathvista_gold(meta: dict) -> str:
    if meta["question_type"] == "multi_choice":
        choices = [str(c).strip() for c in meta["choices"]]
        ans = str(meta["answer"]).strip()
        if ans in choices:
            return chr(ord("A") + choices.index(ans))
        return mathvista_normalize(ans, meta)  # gold given as a letter / "(B)"
    return mathvista_normalize(meta["answer"], meta)


def _missing(a) -> bool:
    return a is None or (isinstance(a, float) and a != a) or str(a).strip() == ""


# ---------------------------------------------------------------- grouping
def group_answers(answers: list[str | None], equiv) -> list[int]:
    """Greedy equivalence grouping in rollout order: an answer joins the first group whose representative
    it matches; None -> NO_ANSWER."""
    reps: list[str] = []
    out = []
    for a in answers:
        if a is None:
            out.append(NO_ANSWER)
            continue
        for gi, rep in enumerate(reps):
            if equiv(rep, a):
                out.append(gi)
                break
        else:
            reps.append(a)
            out.append(len(reps) - 1)
    return out


def grade_pool(df: pd.DataFrame, key: str, cfg) -> pd.DataFrame:
    meta = load_mathvista_meta(cfg["mathvista_metadata"]) if key == "mathvista" else None
    out = []
    for qid, g in df.groupby("qid", sort=True):
        g = g.sort_values("rollout_id")
        if key == "math":
            answers = [None if _missing(a) else str(a) for a in g["extracted"]]
            groups = group_answers(answers, math_equiv)
            gold = str(g["gold"].iloc[0])
            reps = {}
            for a, gi in zip(answers, groups):
                reps.setdefault(gi, a)
            correct = {gi: (gi != NO_ANSWER and math_equiv(gold, rep)) for gi, rep in reps.items()}
            norm = answers
        else:
            m = meta[str(qid)]
            norm = [None if _missing(a) else mathvista_normalize(a, m) for a in g["extracted"]]
            groups = group_answers(norm, lambda x, y: x == y)
            gold = mathvista_gold(m)
            correct = {gi: (gi != NO_ANSWER and a == gold) for a, gi in zip(norm, groups)}
        for rid, a, gi in zip(g["rollout_id"], norm, groups):
            out.append({"qid": qid, "rollout_id": int(rid), "answer_norm": a, "group": gi,
                        "correct": bool(correct[gi])})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    cfg = load_cfg(ap.parse_args().config)
    info = {}
    for key in cfg["models"]:
        od = Path(cfg["paths"]["outputs"]) / key
        df = pd.read_parquet(od / "rollouts.parquet", columns=["qid", "rollout_id", "extracted", "gold"])
        gr = grade_pool(df, key, cfg)
        gr.to_parquet(od / "groups.parquet")
        info[key] = {"rollout_accuracy": float(gr["correct"].mean()),
                     "no_answer_rate": float((gr["group"] == NO_ANSWER).mean()),
                     "mean_distinct_groups": float(gr.groupby("qid")["group"].nunique().mean())}
        print(key, info[key])
    try:
        import math_verify
        info["math_verify_version"] = getattr(math_verify, "__version__", "unknown")
    except ImportError:
        pass
    save_json(info, ensure_dir(cfg["paths"]["results"]) / "grading_summary.json")


if __name__ == "__main__":
    main()
