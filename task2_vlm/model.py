"""SigLIP–Qwen VLM: frozen Qwen3-0.6B (FP16) + trainable FP32 MLP connector (+ LoRA in stages 2–3).

Visual tokens are inserted into the *user* message, before the question, inside Qwen's chat structure:
    <|im_start|>user\\n[196 visual embeddings]\\n{question}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{target}<|im_end|>
Labels cover target tokens and <|im_end|>; visual tokens, prompt and padding are masked (-100).
For the no-image condition the visual tokens are removed and masks/positions are rebuilt (text unchanged).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100
IMG, QPH = "<<IMG>>", "<<Q>>"


class Connector(nn.Module):
    def __init__(self, d_vis=768, d_hidden=1024, d_llm=1024):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_vis, d_hidden), nn.Linear(d_hidden, d_llm)
        self.register_buffer("out_scale", torch.tensor(1.0))   # fixed calibration multiplier (not trained)

    def forward(self, f):
        return self.fc2(F.gelu(self.fc1(f.float()))) * self.out_scale


def pool_2x2(feats: torch.Tensor) -> torch.Tensor:
    """[B,196,D] in 14x14 raster order -> [B,49,D] by averaging each 2x2 spatial block."""
    B, N, D = feats.shape
    x = feats.view(B, 14, 14, D).permute(0, 3, 1, 2)
    return F.avg_pool2d(x.float(), 2).permute(0, 2, 3, 1).reshape(B, 49, D).to(feats.dtype)


class PromptBuilder:
    def __init__(self, tok, caps: dict, enable_thinking: bool = False):
        self.tok, self.caps = tok, caps
        text = tok.apply_chat_template([{"role": "user", "content": f"{IMG}\n{QPH}"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=enable_thinking)
        a, rest = text.split(IMG)
        b, c = rest.split(QPH)
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        self.pre, self.mid, self.post = enc(a), enc(b), enc(c)
        self.eot = tok.convert_tokens_to_ids("<|im_end|>")

    def build(self, question: str, target: str | None, n_vis: int, target_cap: int):
        """Return (ids, labels, vis_mask). Visual slots hold pad ids and are filled with embeddings later."""
        q = self.tok(question, add_special_tokens=False)["input_ids"][: self.caps["question"]]
        prompt = self.pre + [self.tok.pad_token_id] * n_vis + self.mid + q + self.post
        vis = [False] * len(self.pre) + [True] * n_vis + [False] * (len(prompt) - len(self.pre) - n_vis)
        if target is None:
            return prompt, [IGNORE] * len(prompt), vis
        t = self.tok(target, add_special_tokens=False)["input_ids"][:target_cap] + [self.eot]
        return prompt + t, [IGNORE] * len(prompt) + t, vis + [False] * len(t)


def collate(examples, pad_id: int, left_pad: bool = False):
    L = max(len(e[0]) for e in examples)
    B = len(examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), IGNORE, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    vis = torch.zeros((B, L), dtype=torch.bool)
    for i, (x, y, v) in enumerate(examples):
        s = L - len(x) if left_pad else 0
        ids[i, s:s + len(x)] = torch.tensor(x)
        lab[i, s:s + len(y)] = torch.tensor(y)
        att[i, s:s + len(x)] = 1
        vis[i, s:s + len(v)] = torch.tensor(v)
    return ids, lab, att, vis


class VLM(nn.Module):
    def __init__(self, llm, connector: Connector):
        super().__init__()
        self.llm, self.connector = llm, connector

    def embed(self, ids, vis_mask, feats):
        emb = self.llm.get_input_embeddings()(ids)
        if feats is not None and vis_mask.any():
            v = self.connector(feats).to(emb.dtype)                       # [B_img, n_vis, d]
            emb = emb.masked_scatter(vis_mask.unsqueeze(-1), v.reshape(-1, v.shape[-1]))
        return emb

    def hidden(self, ids, att, vis_mask, feats):
        emb = self.embed(ids, vis_mask, feats)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        return self.llm.model(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state

    def per_example_nll(self, ids, lab, att, vis_mask, feats, exclude_eot: int | None = None):
        """Mean token NLL per example over supervised targets (optionally excluding the termination token)."""
        h = self.hidden(ids, att, vis_mask, feats)
        tgt = lab[:, 1:]
        mask = tgt != IGNORE
        if exclude_eot is not None:
            mask &= tgt != exclude_eot
        rows = mask.nonzero(as_tuple=True)
        logits = self.llm.lm_head(h[:, :-1][mask]).float()
        tok_nll = F.cross_entropy(logits, tgt[mask], reduction="none")
        B = ids.shape[0]
        s = torch.zeros(B, device=ids.device).index_add(0, rows[0], tok_nll)
        n = torch.zeros(B, device=ids.device).index_add(0, rows[0], torch.ones_like(tok_nll))
        return s / n.clamp(min=1), n

    @torch.no_grad()
    def greedy(self, ids, att, vis_mask, feats, max_new: int, eot: int, eos: int):
        """Batched greedy decoding with KV cache from left-padded prompts."""
        emb = self.embed(ids, vis_mask, feats)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        out = self.llm(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
        past, B = out.past_key_values, ids.shape[0]
        nxt = out.logits[:, -1].argmax(-1)
        gen, done = [], torch.zeros(B, dtype=torch.bool, device=ids.device)
        cur_pos = pos[:, -1]
        for _ in range(max_new):
            gen.append(torch.where(done, torch.full_like(nxt, eot), nxt))
            done |= (nxt == eot) | (nxt == eos)
            if done.all():
                break
            att = torch.cat([att, torch.ones(B, 1, dtype=att.dtype, device=att.device)], 1)
            cur_pos = cur_pos + 1
            out = self.llm(input_ids=nxt[:, None], attention_mask=att, position_ids=cur_pos[:, None],
                           past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        return torch.stack(gen, 1)


@torch.no_grad()
def calibrate_scale(vlm: VLM, train_feats: torch.Tensor, text_ids: torch.Tensor) -> dict:
    """Rule: scale = mean ||e_text|| over training text tokens / mean ||connector(f)|| over training visual tokens
    (connector at initialisation). Applied as a fixed multiplier on the connector output."""
    vlm.connector.out_scale.fill_(1.0)
    e = vlm.llm.get_input_embeddings()(text_ids.to(vlm.llm.device)).float().norm(dim=-1).mean()
    v = vlm.connector(train_feats.to(next(vlm.connector.parameters()).device)).norm(dim=-1).mean()
    s = float(e / v)
    vlm.connector.out_scale.fill_(s)
    return {"rule": "mean text-embedding norm / mean initial connector-output norm (training data only)",
            "text_embed_norm": float(e), "connector_out_norm_init": float(v), "scale": s}
