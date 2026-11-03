"""Task 2 main run: three-stage curriculum (caption alignment -> visual instruction tuning -> natural VQA).

    python -m task2_vlm.train --config configs/task2.yaml                    # 196 visual tokens
    python -m task2_vlm.train --config configs/task2.yaml --n-vis 49         # optional compression run

Resumable: a full state (connector, adapters, optimizer, scheduler, scaler, sampler RNG + position, torch RNG)
is saved every 250 updates; connector/adapter weights are also saved after each stage.
Loss: token CE averaged within each example, then averaged over the 16 examples of the effective batch.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from common.utils import (append_jsonl, ensure_dir, load_cfg, load_json, peak_mem_gb, read_jsonl, reset_peak_mem,
                          rng_state, save_json, set_rng_state, set_seed, write_jsonl)
from task1_ft.methods import adapter_state_dict, inject_adapters, load_adapter_state
from task2_vlm.common import Features, build_vlm, dev_metrics, load_sets, load_tokenizer, qa_prompt
from task2_vlm.model import PromptBuilder, calibrate_scale, collate


def lr_lambda(total: int, warm_frac: float, min_frac: float):
    warm = max(1, int(round(warm_frac * total)))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return f


class Sampler:
    """Seeded mixture sampler. Stage 1 iterates two shuffled passes over the caption set; later stages draw a
    component per draw (each draw = 2 examples: one complete question pair, or two caption examples)."""

    def __init__(self, sets, cfg, seed):
        self.sets, self.cfg = sets, cfg
        self.rng = np.random.default_rng(seed)
        self.cap_order, self.cap_pos = None, 0
        self.counts = {}

    def state(self):
        return {"rng": self.rng.bit_generator.state, "cap_order": self.cap_order, "cap_pos": self.cap_pos,
                "counts": self.counts}

    def load(self, st):
        self.rng.bit_generator.state = st["rng"]
        self.cap_order, self.cap_pos, self.counts = st["cap_order"], st["cap_pos"], st["counts"]

    def _cap(self, item):
        return {"kind": "caption", "image_id": item["image_id"], "prompt": self.cfg["prompts"]["caption"],
                "target": item["caption"], "cap": self.cfg["caps"]["caption"]}

    def _pair(self, p):
        return [{"kind": p.get("answer_type", p["kind"]), "image_id": it["image_id"],
                 "prompt": qa_prompt(self.cfg, p["question"]), "target": it["answer"], "cap": self.cfg["caps"]["answer"]}
                for it in p["items"]]

    def next_caption_seq(self):
        data = self.sets["caption_train"]
        if self.cap_order is None or self.cap_pos >= len(data):
            self.cap_order, self.cap_pos = self.rng.permutation(len(data)).tolist(), 0
        x = data[self.cap_order[self.cap_pos]]
        self.cap_pos += 1
        return self._cap(x)

    def draw(self, mix: dict) -> list[dict]:
        names = list(mix)
        comp = names[self.rng.choice(len(names), p=np.array([mix[n] for n in names]))]
        self.counts[comp] = self.counts.get(comp, 0) + 1
        if comp == "caption":
            d = self.sets["caption_train"]
            return [self._cap(d[i]) for i in self.rng.integers(0, len(d), 2)]
        pool = {"presence": "presence_train_pool", "counting": "counting_train_pool", "vqa": "vqa_train"}[comp]
        d = self.sets[pool]
        return self._pair(d[self.rng.integers(len(d))])

    def batch(self, stage: dict, n: int) -> list[dict]:
        if stage["name"] == "caption":
            self.counts["caption"] = self.counts.get("caption", 0) + n
            return [self.next_caption_seq() for _ in range(n)]
        out = []
        while len(out) < n:
            out += self.draw(stage["mix"])
        return out


def make_optimizer(vlm, stage, wd):
    groups = []
    conn = [p for p in vlm.connector.parameters() if p.requires_grad]
    dec = [p for p in conn if p.ndim >= 2]
    nodec = [p for p in conn if p.ndim < 2]
    groups += [{"params": dec, "lr": stage["connector_lr"], "weight_decay": wd},
               {"params": nodec, "lr": stage["connector_lr"], "weight_decay": 0.0}]
    if stage["lora_lr"]:
        lora = [p for n, p in vlm.llm.named_parameters() if "lora_" in n]
        groups.append({"params": lora, "lr": stage["lora_lr"], "weight_decay": wd})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--n-vis", type=int, default=None)
    ap.add_argument("--max-updates", type=int, default=None, help="debug only")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    n_vis = args.n_vis or cfg["model"]["n_visual_tokens"]
    run = f"vis{n_vis}"
    tc = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(Path(cfg["paths"]["outputs"]) / run)
    res = ensure_dir(Path(cfg["paths"]["results"]) / run)
    set_seed(cfg["seed"])

    tok = load_tokenizer(cfg["llm"])
    pb = PromptBuilder(tok, cfg["caps"], cfg.get("enable_thinking", False))
    sets = load_sets(cfg)
    feats = Features(cfg, n_vis)
    reset_peak_mem()
    vlm = build_vlm(cfg, device)
    vlm.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.llm.config.use_cache = False

    # connector output calibration (training data only)
    rng = np.random.default_rng(cfg["seed"])
    k = cfg["model"]["calibration_samples"]
    cal_imgs = [sets["caption_train"][i]["image_id"] for i in rng.choice(len(sets["caption_train"]), k, replace=False)]
    cal_text = [tok(c["caption"], add_special_tokens=False)["input_ids"]
                for c in sets["caption_train"][:k]]
    cal = calibrate_scale(vlm, feats.get(cal_imgs), torch.tensor([t for x in cal_text for t in x]))
    save_json(cal, res / "calibration.json")

    stages = cfg["stages"] if "stages" in cfg else tc["stages"]
    bounds = np.cumsum([0] + [s["updates"] for s in stages])
    total = int(bounds[-1])
    sampler = Sampler(sets, cfg, cfg["seed"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    log_path, dev_path = res / "train_log.jsonl", res / "dev_log.jsonl"

    # ---- resume
    start, opt, sched, cur_stage = 0, None, None, -1
    states = sorted(out.glob("state_*"), key=lambda p: int(p.name.split("_")[1]))
    resume = torch.load(states[-1] / "state.pt", weights_only=False) if states else None
    if resume:
        start = resume["update"]
        # the saved optimizer belongs to the stage of the last *completed* update (index start - 1)
        cur_stage = int(np.searchsorted(bounds, start - 1, side="right") - 1)
        if stages[cur_stage]["lora_lr"]:
            torch.manual_seed(cfg["seed"])
            inject_adapters(vlm.llm, "lora", **{k: cfg["model"]["lora"][k] for k in ("r", "alpha", "dropout")},
                            targets=cfg["model"]["lora"]["targets"])
            vlm.llm.to(device)
            load_adapter_state(vlm.llm, resume["adapters"])
        vlm.connector.load_state_dict(resume["connector"])
        opt = make_optimizer(vlm, stages[cur_stage], tc["weight_decay"])
        opt.load_state_dict(resume["opt"])
        st = stages[cur_stage]
        sched = LambdaLR(opt, lr_lambda(st["updates"], tc["warmup_frac"], tc["min_lr_frac"]))
        sched.load_state_dict(resume["sched"])
        scaler.load_state_dict(resume["scaler"])
        sampler.load(resume["sampler"])
        set_rng_state(resume["rng"])
        for p in (log_path, dev_path):
            if p.exists():
                write_jsonl([r for r in read_jsonl(p) if r["update"] <= start], p)
        print("resumed at", start)
    else:
        for p in (log_path, dev_path):
            p.unlink(missing_ok=True)

    timing = load_json(res / "timing.json") if (resume and (res / "timing.json").exists()) else {}
    if start == 0:
        dm = dev_metrics(vlm, pb, feats, sets, cfg, device)
        append_jsonl({"update": 0, "stage": "start", **dm}, dev_path)
        print("dev@0", dm)

    vlm.train()
    end = min(total, args.max_updates or total)
    t_stage = time.perf_counter()
    for u in range(start, end):
        si = int(np.searchsorted(bounds, u, side="right") - 1)
        st = stages[si]
        if si != cur_stage:                                 # stage transition: new param set / optimizer / schedule
            if st["lora_lr"] and not any("lora_" in n for n, _ in vlm.llm.named_parameters()):
                torch.manual_seed(cfg["seed"])
                inject_adapters(vlm.llm, "lora", cfg["model"]["lora"]["r"], cfg["model"]["lora"]["alpha"],
                                cfg["model"]["lora"]["dropout"], cfg["model"]["lora"]["targets"])
                vlm.llm.to(device)
            opt = make_optimizer(vlm, st, tc["weight_decay"])
            sched = LambdaLR(opt, lr_lambda(st["updates"], tc["warmup_frac"], tc["min_lr_frac"]))
            cur_stage = si
            reset_peak_mem()
            t_stage = time.perf_counter()
            save_json({"stage": st["name"],
                       "trainable_connector": sum(p.numel() for p in vlm.connector.parameters() if p.requires_grad),
                       "trainable_lora": sum(p.numel() for n, p in vlm.llm.named_parameters() if p.requires_grad)},
                      res / f"params_{st['name']}.json")
        t0 = time.perf_counter()
        n_eff = tc["microbatch"] * tc["grad_accum"]
        batch = sampler.batch(st, n_eff)
        loss_sum = 0.0
        for m in range(tc["grad_accum"]):
            mb = batch[m * tc["microbatch"]:(m + 1) * tc["microbatch"]]
            ex = [pb.build(x["prompt"], x["target"], n_vis, x["cap"]) for x in mb]
            ids, lab, att, vis = (t.to(device) for t in collate(ex, tok.pad_token_id))
            f = feats.get([x["image_id"] for x in mb]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                nll, _ = vlm.per_example_nll(ids, lab, att, vis, f)
            loss = nll.sum() / n_eff
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g["params"]]
        gn = float(torch.nn.utils.clip_grad_norm_(params, tc["clip"]))
        sc0 = scaler.get_scale()
        lrs = [g["lr"] for g in opt.param_groups]
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        upd = u + 1
        append_jsonl({"update": upd, "stage": st["name"], "loss": loss_sum, "grad_norm": gn,
                      "connector_lr": lrs[0], "lora_lr": lrs[2] if len(lrs) > 2 else None,
                      "skipped": scaler.get_scale() < sc0, "kinds": {k: sum(x["kind"] == k for x in batch)
                                                                    for k in {x["kind"] for x in batch}},
                      "seconds": time.perf_counter() - t0}, log_path)
        if upd % 50 == 0:
            print(f"u{upd} {st['name']} loss {loss_sum:.4f} gn {gn:.3f}", flush=True)

        stage_end = upd in bounds[1:]
        if upd % tc["dev_every"] == 0 or stage_end:
            dm = dev_metrics(vlm, pb, feats, sets, cfg, device)
            append_jsonl({"update": upd, "stage": st["name"], **dm}, dev_path)
            print("dev", upd, {k: round(v, 3) for k, v in dm.items()}, flush=True)
        if stage_end:
            sd = {"connector": vlm.connector.state_dict(), "adapters": adapter_state_dict(vlm.llm)}
            torch.save(sd, out / f"stage{si + 1}_{st['name']}.pt")
            timing[st["name"]] = {"seconds": timing.get(st["name"], {}).get("seconds", 0) + time.perf_counter() - t_stage,
                                  "peak_gpu_mem_gb": peak_mem_gb()}
            save_json(timing, res / "timing.json")
            save_json(sampler.counts, res / f"mixture_counts_through_{st['name']}.json")
        if upd % tc["checkpoint_every"] == 0 or upd == end:
            sd = ensure_dir(out / f"state_{upd:05d}")
            torch.save({"update": upd, "connector": vlm.connector.state_dict(), "adapters": adapter_state_dict(vlm.llm),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "sampler": sampler.state(), "rng": rng_state()}, sd / "state.pt")
            if not stage_end:   # partial-stage time so far (lost on disconnect otherwise)
                timing.setdefault("_partial", {})[st["name"]] = time.perf_counter() - t_stage
                save_json(timing, res / "timing.json")
            for old in sorted(out.glob("state_*"))[:-2]:
                import shutil
                shutil.rmtree(old)
    print("finished", end)


if __name__ == "__main__":
    main()
