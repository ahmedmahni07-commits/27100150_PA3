"""Task 3 Step 4: parallel sampling vs a longer response vs sequential refinement (50 MATH eval questions).

    python -m task3_ttc.live --config configs/task3.yaml generate     # GPU; resumable (one JSONL row per call)
    python -m task3_ttc.live --config configs/task3.yaml summarize

Same 3,072-token output allowance per question:
  single      one response, cap 3,072
  parallel    three independent responses, cap 1,024 each; self-consistency (math-verify groups, ties -> earliest)
  sequential  parallel call #0 reused as the initial answer, then one critique-and-revision call (cap 2,048)
Every call is independently seeded: seed = 6304 * 1000 + 10 * question_index + call_index.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.utils import append_jsonl, ensure_dir, load_cfg, load_json, read_jsonl, save_json
from task3_ttc.grading import NO_ANSWER, group_answers, math_equiv
from task3_ttc.selection import vote_winner

CALLS = {"single": 0, "par0": 1, "par1": 2, "par2": 3, "revise": 4}


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) using balanced-brace matching; None if absent/unbalanced."""
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    i = text.index("{", idx)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None


def load_problems(cfg) -> pd.DataFrame:
    split = load_json(Path(cfg["paths"]["results"]) / "splits.json")["math"]["live_math"]
    df = pd.read_parquet(Path(cfg["paths"]["outputs"]) / "math" / "rollouts.parquet",
                         columns=["qid", "rollout_id", "problem", "gold"])
    df = df[df["rollout_id"] == df["rollout_id"].min()].set_index("qid")
    if df["problem"].isna().any():
        raise ValueError("problem text not in generations.parquet – add its column name to configs/task3.yaml")
    return df.loc[split].reset_index()


class Generator:
    def __init__(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        lc = cfg["live"]
        self.cfg = lc
        self.tok = AutoTokenizer.from_pretrained(lc["model"], revision=lc["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(lc["model"], revision=lc["revision"],
                                                          dtype=getattr(torch, lc["dtype"])).cuda().eval()

    def messages(self, problem: str) -> list[dict]:
        return [{"role": "system", "content": self.cfg["system"]},
                {"role": "user", "content": f"{problem}\n\n{self.cfg['instruction']}"}]

    @torch.no_grad()
    def __call__(self, msgs: list[dict], cap: int, seed: int) -> dict:
        lc = self.cfg
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=lc["temperature"],
                                  top_p=lc["top_p"], top_k=lc["top_k"], repetition_penalty=lc["repetition_penalty"],
                                  pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, enc["input_ids"].shape[1]:]
        eos_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        ended = any(int(t) in eos_ids for t in gen)
        n_out = int(len(gen)) - (1 if ended else 0)
        resp = self.tok.decode(gen, skip_special_tokens=True)
        return {"prompt": text, "response": resp, "input_tokens": int(enc["input_ids"].shape[1]),
                "output_tokens": n_out, "truncated": (not ended) and len(gen) >= cap, "seconds": dt,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3, "seed": seed,
                "extracted": last_boxed(resp)}


def generate(cfg):
    lc = cfg["live"]
    out = ensure_dir(Path(cfg["paths"]["results"]) / "live")
    path = out / "calls.jsonl"
    done = {(r["qid"], r["call"]) for r in read_jsonl(path)} if path.exists() else set()
    probs = load_problems(cfg)
    gen = Generator(cfg)
    for qi, row in probs.iterrows():
        base = gen.messages(row["problem"])
        jobs = [("single", base, lc["single_cap"])] + [(f"par{j}", base, lc["parallel_cap"]) for j in range(lc["n_parallel"])]
        for name, msgs, cap in jobs:
            if (row["qid"], name) in done:
                continue
            r = gen(msgs, cap, cfg["seed"] * 1000 + 10 * qi + CALLS[name])
            append_jsonl({"qid": row["qid"], "q_index": qi, "call": name, "cap": cap, **r}, path)
        if (row["qid"], "revise") not in done:
            init = next(r for r in read_jsonl(path) if r["qid"] == row["qid"] and r["call"] == "par0")
            msgs = base + [{"role": "assistant", "content": init["response"]},
                           {"role": "user", "content": lc["revise_instruction"]}]
            r = gen(msgs, lc["revise_cap"], cfg["seed"] * 1000 + 10 * qi + CALLS["revise"])
            append_jsonl({"qid": row["qid"], "q_index": qi, "call": "revise", "cap": lc["revise_cap"], **r}, path)
        print(qi, row["qid"], "done", flush=True)


def summarize(cfg):
    out = Path(cfg["paths"]["results"]) / "live"
    calls = pd.DataFrame(read_jsonl(out / "calls.jsonl"))
    gold = load_problems(cfg).set_index("qid")["gold"]
    corr = lambda a: a is not None and math_equiv(str(gold_q), a)
    rows, trans = [], []
    for qid, g in calls.groupby("qid"):
        c = g.set_index("call")
        gold_q = str(gold[qid])
        par = [c.loc[f"par{j}"] for j in range(3)]
        groups = group_answers([p["extracted"] for p in par], math_equiv)
        win, _ = vote_winner(np.array(groups), np.ones(3))
        sc_ans = None if win == NO_ANSWER else par[groups.index(win)]["extracted"]
        strat = {
            "single": ([c.loc["single"]], c.loc["single"]["extracted"]),
            "parallel": (par, sc_ans),
            "sequential": ([par[0], c.loc["revise"]], c.loc["revise"]["extracted"]),
        }
        for s, (used, ans) in strat.items():
            rows.append({"qid": qid, "strategy": s, "correct": corr(ans), "answer": ans,
                         "input_tokens": sum(int(u["input_tokens"]) for u in used),
                         "output_tokens": sum(int(u["output_tokens"]) for u in used),
                         "seconds": sum(float(u["seconds"]) for u in used),
                         "any_truncated": any(bool(u["truncated"]) for u in used),
                         "final_truncated": bool(used[-1]["truncated"]),
                         "extraction_fail": ans is None, "peak_mem_gb": max(float(u["peak_mem_gb"]) for u in used)})
        ic, rc = corr(par[0]["extracted"]), corr(c.loc["revise"]["extracted"])
        trans.append({"qid": qid, "initial_correct": ic, "revised_correct": rc,
                      "answer_changed": par[0]["extracted"] != c.loc["revise"]["extracted"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_question.csv", index=False)
    summ = df.groupby("strategy").agg(acc=("correct", "mean"), input_tokens=("input_tokens", "mean"),
                                      output_tokens=("output_tokens", "mean"), seconds=("seconds", "mean"),
                                      trunc_rate=("final_truncated", "mean"), any_trunc_rate=("any_truncated", "mean"),
                                      extraction_fail=("extraction_fail", "mean"), peak_mem_gb=("peak_mem_gb", "max"))
    for c in ["acc", "trunc_rate", "any_trunc_rate", "extraction_fail"]:
        summ[c] *= 100
    summ.to_csv(out / "summary.csv")
    t = pd.DataFrame(trans)
    tr = {"wrong_to_correct": int((~t.initial_correct & t.revised_correct).sum()),
          "correct_to_wrong": int((t.initial_correct & ~t.revised_correct).sum()),
          "correct_to_correct": int((t.initial_correct & t.revised_correct).sum()),
          "wrong_to_wrong": int((~t.initial_correct & ~t.revised_correct).sum()),
          "answer_changed": int(t.answer_changed.sum())}
    t.to_csv(out / "transitions_per_question.csv", index=False)
    save_json(tr, out / "transitions.json")
    print(summ.round(2).to_string(), "\n", tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("step", choices=["generate", "summarize"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    generate(cfg) if args.step == "generate" else summarize(cfg)


if __name__ == "__main__":
    main()
