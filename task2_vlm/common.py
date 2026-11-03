"""Shared Task 2 helpers: loading the model, features, example construction, and teacher-forced dev metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common.utils import load_json
from task2_vlm.model import VLM, Connector, PromptBuilder, collate, pool_2x2


def load_tokenizer(name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_vlm(cfg, device):
    from transformers import AutoModelForCausalLM

    llm = AutoModelForCausalLM.from_pretrained(cfg["llm"], dtype=torch.float16, attn_implementation="sdpa").to(device)
    for p in llm.parameters():
        p.requires_grad_(False)
    conn = Connector(768, cfg["model"]["connector_hidden"], llm.config.hidden_size).to(device)
    return VLM(llm, conn)


class Features:
    def __init__(self, cfg, n_vis: int):
        blob = torch.load(Path(cfg["paths"]["outputs"]) / "features.pt")
        self.feats, self.index, self.n_vis = blob["feats"], blob["index"], n_vis

    def get(self, image_ids) -> torch.Tensor:
        f = self.feats[[self.index[i] for i in image_ids]]
        return pool_2x2(f) if self.n_vis == 49 else f


def load_sets(cfg) -> dict:
    sd = Path(cfg["data"]["split_dir"])
    names = ["caption_train", "caption_dev", "presence_train_pool", "counting_train_pool", "vqa_train",
             "grounding_dev", "grounding_test", "vqa_dev", "vqa_test"]
    return {n: load_json(sd / f"{n}.json") for n in names}


def qa_prompt(cfg, question: str) -> str:
    return f"{question}\n{cfg['prompts']['short_answer_suffix']}"


def flatten_pairs(pairs) -> list[dict]:
    """One row per question instance with its partner image (complementary swap)."""
    rows = []
    for p in pairs:
        for j, it in enumerate(p["items"]):
            partner = p["items"][1 - j]
            rows.append({"pair_id": p["pair_id"], "slot": j, "kind": p.get("answer_type", p["kind"]),
                         "question": p["question"], "image_id": it["image_id"], "partner_image_id": partner["image_id"],
                         "answer": it["answer"], "partner_answer": partner["answer"],
                         "answers": it.get("answers", [it["answer"]])})
    return rows


def mismatch_map(rows: list[dict], seed: int) -> dict:
    """Fixed unrelated-image mapping: each question gets an image from a *different* pair of the same suite
    (never either image of its own pair). Pairs are deranged with a seeded permutation."""
    pids = sorted({r["pair_id"] for r in rows})
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(pids))
        if all(perm[i] != i for i in range(len(pids))):
            break
    to = {pids[i]: pids[perm[i]] for i in range(len(pids))}
    img = {(r["pair_id"], r["slot"]): r["image_id"] for r in rows}
    own = {}
    for r in rows:
        own.setdefault(r["pair_id"], set()).add(r["image_id"])
    out = {}
    for r in rows:
        m = img[(to[r["pair_id"]], r["slot"])]
        assert m not in own[r["pair_id"]]
        out[f"{r['pair_id']}|{r['slot']}"] = m
    return out


@torch.no_grad()
def teacher_forced_nll(vlm, pb: PromptBuilder, feats: Features, rows, image_key: str | None, cfg, device,
                       prompt_fn, target_cap: int, exclude_eot: bool, bs: int = 8) -> np.ndarray:
    """Per-row mean token NLL of the reference (row['answer'] or row['caption']) given the chosen image."""
    out = []
    n_vis = feats.n_vis if image_key else 0
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        ex = [pb.build(prompt_fn(r), r.get("answer", r.get("caption")), n_vis, target_cap) for r in chunk]
        ids, lab, att, vis = (t.to(device) for t in collate(ex, pb.tok.pad_token_id))
        f = feats.get([r[image_key] for r in chunk]).to(device) if image_key else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            nll, _ = vlm.per_example_nll(ids, lab, att, vis, f, exclude_eot=pb.eot if exclude_eot else None)
        out.append(nll.float().cpu())
    return torch.cat(out).numpy()


def dev_metrics(vlm, pb, feats, sets, cfg, device) -> dict:
    vlm.eval()
    caps = cfg["caps"]
    m = {}
    cap_rows = [{"caption": c["caption"], "image_id": c["image_id"]} for c in sets["caption_dev"]]
    m["dev_caption_loss"] = float(teacher_forced_nll(vlm, pb, feats, cap_rows, "image_id", cfg, device,
                                                     lambda r: cfg["prompts"]["caption"], caps["caption"], False).mean())
    for suite in ["grounding_dev", "vqa_dev"]:
        rows = flatten_pairs(sets[suite])
        res = {}
        for cond, key in [("correct", "image_id"), ("swap", "partner_image_id"), ("none", None)]:
            res[cond] = teacher_forced_nll(vlm, pb, feats, rows, key, cfg, device, lambda r: qa_prompt(cfg, r["question"]),
                                           caps["answer"], True)
            m[f"{suite}_nll_{cond}"] = float(res[cond].mean())
        m[f"{suite}_swap_penalty"] = float((res["swap"] - res["correct"]).mean())
        m[f"{suite}_noimg_penalty"] = float((res["none"] - res["correct"]).mean())
    vlm.train()
    return m
