"""Task 1 data: MetaMathQA selection, chat formatting and response-only label masking.

Selection rule (handout): walk MetaMathQA in source order and keep the first 40,000 *eligible* rows.
A row is ineligible if (a) its query or response is empty, (b) its formatted length (prompt + response +
end-of-turn) exceeds 2,048 tokens, or (c) its `query` or `original_question` matches a GSM8K *test* question
after text normalisation. Retained rows are never truncated or shuffled.

    python -m task1_ft.data --config configs/task1.yaml      # writes data/task1/{selection.json, train_ids.json}
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import torch

from common.utils import ensure_dir, load_cfg, save_json, load_json

IGNORE = -100


def normalize_text(s: str | None) -> str:
    """Normalisation used for GSM8K-test decontamination: NFKC, lower-case, collapse whitespace, strip."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def render_prompt(tok, question: str, system: str | None = None, enable_thinking: bool = False) -> str:
    """Chat-formatted prompt ending with the assistant header (identical string at train and test time).

    With enable_thinking=False the Qwen3 template appends an empty `<think>\\n\\n</think>\\n\\n` block, so the
    supervised response starts exactly where generation starts.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def end_of_turn_token(tok) -> str:
    return "<|im_end|>"


def encode_example(tok, question: str, response: str, enable_thinking: bool = False) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels). Labels are -100 on the prompt; the response and <|im_end|> are supervised."""
    p_ids = tok(render_prompt(tok, question, enable_thinking=enable_thinking), add_special_tokens=False)["input_ids"]
    r_ids = tok(response + end_of_turn_token(tok), add_special_tokens=False)["input_ids"]
    return p_ids + r_ids, [IGNORE] * len(p_ids) + r_ids


def gsm8k_test_questions(cfg) -> set[str]:
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["gsm8k_dataset"], "main", split="test")
    return {normalize_text(q) for q in ds["question"]}


def select_examples(cfg, tok) -> dict:
    """Scan MetaMathQA in source order; return selection dict with source indices and exclusion counts."""
    from datasets import load_dataset

    n_target, max_len = cfg["data"]["n_train"], cfg["data"]["max_len"]
    think = cfg.get("enable_thinking", False)
    ds = load_dataset(cfg["data"]["train_dataset"], split="train")
    test_q = gsm8k_test_questions(cfg)

    kept, reasons = [], {"empty": 0, "too_long": 0, "gsm8k_test_overlap": 0}
    chunk = 4096
    for start in range(0, len(ds), chunk):
        rows = ds[start : start + chunk]
        for j in range(len(rows["query"])):
            q, r = rows["query"][j], rows["response"][j]
            oq = rows.get("original_question", [None] * len(rows["query"]))[j]
            if not q or not q.strip() or not r or not r.strip():
                reasons["empty"] += 1
                continue
            if normalize_text(q) in test_q or (oq and normalize_text(oq) in test_q):
                reasons["gsm8k_test_overlap"] += 1
                continue
            ids, _ = encode_example(tok, q, r, think)
            if len(ids) > max_len:
                reasons["too_long"] += 1
                continue
            kept.append(start + j)
            if len(kept) == n_target:
                break
        if len(kept) == n_target:
            break
    scanned = kept[-1] + 1
    return {"source_indices": kept, "n_scanned": scanned, "excluded": reasons,
            "normalisation": "NFKC + lowercase + collapse whitespace + strip"}


class SFTData:
    """Tokenised training set in fixed source order. Cached to outputs/ as a .pt file."""

    def __init__(self, cfg, tok):
        split_dir = ensure_dir(cfg["data"]["split_dir"])
        sel_path = split_dir / "selection.json"
        cache = Path(cfg["paths"]["outputs"]) / "train_tokens.pt"
        if cache.exists():
            blob = torch.load(cache)
            self.input_ids, self.labels, self.source_indices = blob["input_ids"], blob["labels"], blob["source_indices"]
            return
        if sel_path.exists():
            sel = load_json(sel_path)
        else:
            sel = select_examples(cfg, tok)
            save_json(sel, sel_path)
        from datasets import load_dataset

        ds = load_dataset(cfg["data"]["train_dataset"], split="train").select(sel["source_indices"])
        think = cfg.get("enable_thinking", False)
        self.input_ids, self.labels = [], []
        for q, r in zip(ds["query"], ds["response"]):
            ids, lab = encode_example(tok, q, r, think)
            self.input_ids.append(torch.tensor(ids, dtype=torch.int32))
            self.labels.append(torch.tensor(lab, dtype=torch.int32))
        self.source_indices = sel["source_indices"]
        ensure_dir(cache.parent)
        torch.save({"input_ids": self.input_ids, "labels": self.labels, "source_indices": self.source_indices}, cache)

    def __len__(self):
        return len(self.input_ids)

    def microbatch(self, start: int, size: int, pad_id: int):
        """Right-padded microbatch of examples [start, start+size) in source order."""
        ids = self.input_ids[start : start + size]
        labs = self.labels[start : start + size]
        return collate(ids, labs, pad_id)


def collate(ids_list, labs_list, pad_id: int):
    L = max(len(x) for x in ids_list)
    B = len(ids_list)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids_list, labs_list)):
        input_ids[i, : len(x)] = x.long()
        labels[i, : len(y)] = y.long()
        attn[i, : len(x)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def n_target_tokens(labels: torch.Tensor) -> int:
    """Number of supervised next-token targets (labels shifted by one)."""
    return int((labels[:, 1:] != IGNORE).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    tok = load_tokenizer(cfg["model"])
    data = SFTData(cfg, tok)
    lens = [len(x) for x in data.input_ids]
    resp = [int((y != IGNORE).sum()) for y in data.labels]
    stats = {"n": len(data), "mean_len": sum(lens) / len(lens), "max_len": max(lens),
             "total_tokens": sum(lens), "supervised_tokens": sum(resp)}
    save_json(stats, Path(cfg["data"]["split_dir"]) / "train_stats.json")
    print(stats)
