"""Task 1 Step 7: greedy responses to the 24 first_plot_questions prompts for all 7 evaluation conditions.

    python -m task1_ft.qualitative --config configs/task1.yaml            # all conditions whose model exists
    python -m task1_ft.qualitative --config configs/task1.yaml --annotation-sheet

Writes results/task1/qualitative/responses.jsonl (168 rows when complete) and, with --annotation-sheet,
annotation_sheet.csv: the fixed 42-response subset (first and last plain questions x {plain, json, template}
x 7 conditions) with EMPTY columns for you to fill in by hand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from common.utils import ensure_dir, load_cfg, read_jsonl, set_seed, write_jsonl
from task1_ft.data import load_tokenizer, render_prompt
from task1_ft.evaluate import build_eval_model_dir, cond_name

CONDITIONS = [("baseline", 0), ("full_ft", 625), ("full_ft", 1250), ("l2sp", 1250), ("lora4", 1250),
              ("lora16", 1250), ("oplora16", 1250)]


def load_questions(path: str) -> list[dict]:
    qs = yaml.safe_load(open(path))
    out = []
    for q in qs:
        fmt = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "plain"
        base = q["id"].removesuffix("_json").removesuffix("_template")
        out.append({"qid": q["id"], "base_id": base, "format": fmt, "system": q.get("system"),
                    "prompt": q["paraphrases"][0]})
    assert len(out) == 24, len(out)
    return out


@torch.no_grad()
def generate_all(path: str, tok, questions, max_new: int, device: str) -> list[dict]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
    rows = []
    for q in questions:
        text = render_prompt(tok, q["prompt"], system=q["system"], enable_thinking=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        gen = out[0, enc["input_ids"].shape[1]:]
        ended = bool((gen == tok.convert_tokens_to_ids("<|im_end|>")).any() or (gen == tok.eos_token_id).any())
        rows.append({**q, "response": tok.decode(gen, skip_special_tokens=True), "n_tokens": int(len(gen)),
                     "truncated": (not ended) and len(gen) >= max_new})
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--annotation-sheet", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    res = ensure_dir(Path(cfg["paths"]["results"]) / "qualitative")
    out_path = res / "responses.jsonl"
    qs = load_questions(cfg["qualitative"]["questions_file"])
    done = read_jsonl(out_path) if out_path.exists() else []
    have = {r["condition"] for r in done}
    tok = load_tokenizer(cfg["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.annotation_sheet:
        for method, upd in CONDITIONS:
            name = cond_name(method, upd)
            if name in have:
                continue
            try:
                path, _ = build_eval_model_dir(cfg, method, upd, device)
            except FileNotFoundError as e:
                print("skip", name, e)
                continue
            set_seed(cfg["seed"])
            rows = generate_all(path, tok, qs, cfg["qualitative"]["max_new_tokens"], device)
            done += [{"condition": name, **r} for r in rows]
            write_jsonl(done, out_path)
            print(name, "done")

    # grouped view: question -> format -> condition
    df = pd.DataFrame(done)
    if len(df):
        order = [cond_name(m, u) for m, u in CONDITIONS]
        df["cond_order"] = df["condition"].map({c: i for i, c in enumerate(order)})
        df.sort_values(["base_id", "format", "cond_order"]).drop(columns="cond_order").to_csv(
            res / "responses_grouped.csv", index=False)
        plain_order = [q["base_id"] for q in qs if q["format"] == "plain"]
        first, last = plain_order[0], plain_order[-1]
        sub = df[df["base_id"].isin([first, last])].sort_values(["base_id", "format", "cond_order"])
        sheet = sub[["condition", "base_id", "format", "prompt", "response", "n_tokens", "truncated"]].copy()
        for col in ["format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing",
                    "inappropriate_content", "notes"]:
            sheet[col] = ""
        if args.annotation_sheet:
            sheet.to_csv(res / "annotation_sheet.csv", index=False)
            print(f"annotation sheet: {len(sheet)} rows ({first}, {last})")


if __name__ == "__main__":
    main()
