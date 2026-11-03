"""Shared training loop for the five Task 1 variants (Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16).

    python -m task1_ft.train --config configs/task1.yaml --method full_ft
    python -m task1_ft.train --config configs/task1.yaml --method l2sp
    python -m task1_ft.train --config configs/task1.yaml --method lora4    # lora16, oplora16

Re-running the same command resumes from the newest saved state (model/adapters, optimizer, scheduler,
grad-scaler, RNG and data position).

Loss: next-token cross-entropy on assistant-response tokens + <|im_end|> only, summed over all valid response
tokens of the 32-example effective batch and divided by that count (so every token in the effective batch has
equal weight, independent of how they fall into microbatches). For L2-SP, lambda * Omega is added once per
optimizer update; SFT loss and the total objective are logged separately.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.utils import (append_jsonl, dir_size_mb, ensure_dir, load_cfg, load_json, peak_mem_gb,
                          reset_peak_mem, rng_state, save_json, set_rng_state, set_seed)
from task1_ft.data import IGNORE, SFTData, load_tokenizer, n_target_tokens
from task1_ft.methods import (L2SPReference, adapter_state_dict, build_model, export_effective_factors,
                              load_adapter_state, param_groups)

CE_CHUNK = 1024


def response_ce_sum(model, batch, device) -> torch.Tensor:
    """Sum of token CE over supervised positions. Logits are only formed for supervised rows, in
    checkpointed chunks, so a 2 x 2048 microbatch never materialises a 2 x 2048 x 151k logit tensor."""
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    hidden = model.model(input_ids=ids, attention_mask=attn).last_hidden_state  # B x L x H
    tgt = labels[:, 1:]
    mask = tgt != IGNORE
    h = hidden[:, :-1][mask]
    y = tgt[mask]
    total = hidden.new_zeros((), dtype=torch.float32)

    def chunk_loss(hc, yc):
        return F.cross_entropy(model.lm_head(hc).float(), yc, reduction="sum")

    for s in range(0, h.shape[0], CE_CHUNK):
        total = total + checkpoint(chunk_loss, h[s:s + CE_CHUNK], y[s:s + CE_CHUNK], use_reentrant=False)
    return total


def _atomic_save(obj, path: Path):
    """Write to a temp file, flush to disk, then rename: a disconnect never leaves a half-written checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_state(out: Path, update: int, model, info, opt, sched, scaler, is_full: bool, keep: int,
               save_weights: bool):
    """Resume state at `update`; full weights additionally kept for evaluation stages.
    A state counts as valid only once its COMPLETE marker exists (written last)."""
    st_dir = ensure_dir(out / f"state_{update:04d}")
    (st_dir / "COMPLETE").unlink(missing_ok=True)
    sizes = {}
    if is_full:
        _atomic_save(model.state_dict(), st_dir / "model_fp32.pt")
    else:
        _atomic_save(adapter_state_dict(model), st_dir / "adapters.pt")
    _atomic_save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                  "rng": rng_state(), "update": update, "examples_seen": update * 32}, st_dir / "trainer.pt")
    (st_dir / "COMPLETE").write_text(str(update))
    sizes["resume_state_mb"] = dir_size_mb(st_dir)
    if save_weights:
        w_dir = ensure_dir(out / f"weights_{update:04d}")
        if is_full:
            # FP32 master weights: needed for exact dW = W_trained - W0 diagnostics; fp16 copy is made at eval time
            _atomic_save(model.state_dict(), w_dir / "model_fp32.pt")
        else:
            _atomic_save(adapter_state_dict(model), w_dir / "adapters.pt")
            _atomic_save(export_effective_factors(model), w_dir / "effective_factors.pt")
        sizes["weights_mb"] = dir_size_mb(w_dir)
    # prune older resume states (weights_* are kept); never prune below `keep` valid states
    states = [d for d in sorted(out.glob("state_*")) if (d / "COMPLETE").exists()]
    for old in states[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return sizes


def _loads(path: Path) -> bool:
    try:
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return True
    except Exception as e:  # corrupted / truncated file
        print(f"skipping unreadable checkpoint {path}: {str(e)[:120]}")
        return False


def latest_state(out: Path, is_full: bool):
    """Newest state whose files all load; corrupted states (e.g. from a disconnect mid-save) are skipped."""
    states = sorted(out.glob("state_*"), key=lambda p: int(p.name.split("_")[1]))
    for s in reversed(states):
        files = [s / "trainer.pt", s / ("model_fp32.pt" if is_full else "adapters.pt")]
        if all(f.exists() for f in files) and all(_loads(f) for f in files):
            return s
        print(f"ignoring incomplete/corrupted state {s.name}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--max-updates", type=int, default=None, help="debug: stop early")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    tc, mc = cfg["train"], cfg["methods"][args.method]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(Path(cfg["paths"]["outputs"]) / args.method)
    res = ensure_dir(Path(cfg["paths"]["results"]) / "train" / args.method)
    log_path = res / "log.jsonl"

    set_seed(cfg["seed"])
    tok = load_tokenizer(cfg["model"])
    data = SFTData(cfg, tok)
    eff = tc["microbatch"] * tc["grad_accum"]
    assert len(data) >= tc["total_updates"] * eff, "not enough examples for the update budget"

    set_seed(cfg["seed"])  # identical adapter init across LoRA-16 / OPLoRA-16
    reset_peak_mem()
    model, info = build_model(cfg, args.method, device)
    is_full = mc["kind"] == "full"
    if tc["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    lam = float(mc.get("lam", 0.0))
    l2sp = None
    if lam > 0:
        l2sp = L2SPReference(model, device=mc.get("ref_device", device) if device == "cuda" else "cpu",
                             dtype=getattr(torch, mc.get("ref_dtype", "bfloat16")))
        info.update({"l2sp_ref_storage_mb": l2sp.storage_mb, "l2sp_ref_setup_seconds": l2sp.setup_seconds,
                     "l2sp_ref_params": l2sp.n_params, "l2sp_ref_roundtrip_max_err": l2sp.max_roundtrip_err,
                     "l2sp_ref_device": mc.get("ref_device", device)})
    # displacement reference for Full FT (CPU copy, used only at checkpoints; not part of the objective)
    disp_ref = l2sp if l2sp is not None else (L2SPReference(model, device="cpu") if is_full else None)

    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=mc["lr"], betas=tuple(tc["betas"]),
                            eps=tc["eps"])
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, tc["warmup_updates"], tc["total_updates"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    start_update = 0
    st = latest_state(out, is_full)
    prev_time = 0.0
    if st is not None:
        tr = torch.load(st / "trainer.pt", weights_only=False)
        if is_full:
            model.load_state_dict(torch.load(st / "model_fp32.pt"))
        else:
            load_adapter_state(model, torch.load(st / "adapters.pt"))
        opt.load_state_dict(tr["opt"])
        sched.load_state_dict(tr["sched"])
        scaler.load_state_dict(tr["scaler"])
        set_rng_state(tr["rng"])
        start_update = tr["update"]
        prev_time = load_json(res / "summary.json").get("train_seconds", 0.0) if (res / "summary.json").exists() else 0.0
        if log_path.exists():  # drop log rows written after the resumed state by the interrupted run
            from common.utils import read_jsonl, write_jsonl
            write_jsonl([r for r in read_jsonl(log_path) if r["update"] <= start_update], log_path)
        print(f"resumed from {st} at update {start_update}")
    else:
        log_path.unlink(missing_ok=True)
    save_json(info, res / "model_info.json")

    end_update = min(tc["total_updates"], args.max_updates or tc["total_updates"])
    ckpt_sizes, skipped = {}, 0
    if (res / "summary.json").exists() and st is not None:
        prev = load_json(res / "summary.json")
        ckpt_sizes, skipped = prev.get("checkpoint_sizes_mb", {}), prev.get("skipped_updates", 0)
    t_start = time.perf_counter()
    for update in range(start_update, end_update):
        t0 = time.perf_counter()
        base = update * eff
        micro = [data.microbatch(base + i * tc["microbatch"], tc["microbatch"], tok.pad_token_id)
                 for i in range(tc["grad_accum"])]
        n_tok = sum(n_target_tokens(mb["labels"]) for mb in micro)
        sft_sum = 0.0
        omega_val = None
        for i, mb in enumerate(micro):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                ce = response_ce_sum(model, mb, device)
            loss = ce / n_tok
            sft_sum += float(ce.detach())
            if l2sp is not None and i == len(micro) - 1:
                omega = l2sp.omega(model)               # FP32, outside autocast
                omega_val = float(omega.detach())
                loss = loss + lam * omega
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["clip"]))
        scale_before = scaler.get_scale()
        lr = sched.get_last_lr()[0]
        scaler.step(opt)
        scaler.update()
        was_skipped = scaler.get_scale() < scale_before
        skipped += int(was_skipped)
        sched.step()
        opt.zero_grad(set_to_none=True)

        sft = sft_sum / n_tok
        row = {"update": update + 1, "examples_seen": (update + 1) * eff, "sft_loss": sft, "lr": lr,
               "grad_norm": gnorm, "loss_scale": scale_before, "skipped": bool(was_skipped),
               "n_tokens": n_tok, "step_seconds": time.perf_counter() - t0}
        if omega_val is not None:
            row.update({"omega": omega_val, "lam_omega": lam * omega_val, "total_loss": sft + lam * omega_val})
        append_jsonl(row, log_path)
        if (update + 1) % 25 == 0 or update == start_update:
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        u = update + 1
        if u in tc["checkpoint_updates"]:
            sizes = save_state(out, u, model, info, opt, sched, scaler, is_full, tc["keep_resume_states"],
                               save_weights=u in tc["eval_updates"])
            ckpt_sizes[str(u)] = sizes
            extra = {}
            if disp_ref is not None:
                # ||theta - theta0||_2 over unique parameters, against the exact bf16 starting point
                extra[f"displacement_{u}"] = disp_ref.displacement(model)
            summ = {"method": args.method, "last_update": u, "train_seconds": prev_time + time.perf_counter() - t_start,
                    "peak_gpu_mem_gb": peak_mem_gb(), "checkpoint_sizes_mb": ckpt_sizes, "skipped_updates": skipped,
                    **({"l2sp_penalty_seconds": l2sp.penalty_seconds} if l2sp else {}), **extra}
            old = load_json(res / "summary.json") if (res / "summary.json").exists() else {}
            old.update(summ)
            save_json(old, res / "summary.json")
    print("done:", load_json(res / "summary.json") if (res / "summary.json").exists() else {})


if __name__ == "__main__":
    main()
