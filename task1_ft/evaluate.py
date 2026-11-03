"""Task 1 evaluation: build an FP16 evaluation model for a condition and score it with lm-evaluation-harness.

Conditions (7): baseline, full_ft@625 (midpoint), full_ft@1250, l2sp@1250, lora4@1250, lora16@1250, oplora16@1250.

    python -m task1_ft.evaluate --config configs/task1.yaml --condition baseline
    python -m task1_ft.evaluate --config configs/task1.yaml --condition full_ft --update 625
    python -m task1_ft.evaluate --config configs/task1.yaml --condition oplora16           # final (1250)

Adapters are merged into a *fresh* FP16 backbone copy (OPLoRA: projected factors B~, A~). Before merging,
merged-vs-unmerged logits are compared on a small batch (FP32) and the max |diff| is saved.
Outputs: results/task1/eval/<name>/{scores.json, samples_<task>.jsonl, merge_check.json}
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from common.utils import ensure_dir, load_cfg, save_json, set_seed, write_jsonl
from task1_ft.data import load_tokenizer, render_prompt
from task1_ft.methods import inject_adapters, load_adapter_state, merge_effective_factors


def cond_name(condition: str, update: int) -> str:
    return "baseline" if condition == "baseline" else f"{condition}_{update}"


def build_eval_model_dir(cfg, condition: str, update: int, device: str) -> tuple[str, dict]:
    """Return a path/HF-id loadable by lm-eval and a dict describing how it was built."""
    if condition == "baseline":
        return cfg["model"], {"source": "original checkpoint"}
    from transformers import AutoModelForCausalLM

    out = Path(cfg["paths"]["outputs"])
    wdir = out / condition / f"weights_{update:04d}"
    merged_dir = out / "merged" / cond_name(condition, update)
    meta = {"weights_dir": str(wdir), "merged_dir": str(merged_dir)}
    if (merged_dir / "config.json").exists():
        return str(merged_dir), meta
    mc = cfg["methods"][condition]
    tok = load_tokenizer(cfg["model"])
    if mc["kind"] == "full":
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
        model.load_state_dict(torch.load(wdir / "model_fp32.pt", map_location="cpu"))
        model = model.to(torch.float16)
    else:
        meta["merge_check"] = merge_check(cfg, condition, wdir, tok, device)
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16)
        factors = torch.load(wdir / "effective_factors.pt", map_location="cpu")
        meta["n_merged"] = merge_effective_factors(model, factors)
    ensure_dir(merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    del model
    gc.collect()
    return str(merged_dir), meta


@torch.no_grad()
def merge_check(cfg, condition: str, wdir: Path, tok, device: str) -> dict:
    """Compare logits of the unmerged adapter model and the merged model on 4 short prompts (FP32)."""
    from transformers import AutoModelForCausalLM

    mc, lc = cfg["methods"][condition], cfg["lora_common"]
    prompts = [render_prompt(tok, q) + "Let's" for q in
               ["What is 17 * 23?", "Name a primary colour.", "Solve x + 3 = 10.", "hey I feel bored"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    m1 = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(device).eval()
    inject_adapters(m1, mc["kind"], mc["r"], mc["alpha"], 0.0, lc["targets"], mc.get("k"))
    load_adapter_state(m1, torch.load(wdir / "adapters.pt", map_location=device))
    l1 = m1(**enc).logits.float()
    del m1
    m2 = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    merge_effective_factors(m2, torch.load(wdir / "effective_factors.pt", map_location="cpu"))
    m2 = m2.to(device).eval()
    l2 = m2(**enc).logits.float()
    del m2
    mask = enc["attention_mask"].bool()
    diff = (l1 - l2).abs()[mask]
    return {"max_abs_logit_diff_fp32": float(diff.max()), "mean_abs_logit_diff_fp32": float(diff.mean()),
            "argmax_agreement": float((l1.argmax(-1) == l2.argmax(-1))[mask].float().mean())}


def make_lm(path: str, cfg, batch_size):
    from lm_eval.models.huggingface import HFLM

    class HFLMNoThink(HFLM):
        """Same as HFLM but renders the chat template with enable_thinking=False (Qwen3)."""

        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt, enable_thinking=False)

    return HFLMNoThink(pretrained=path, dtype=cfg["eval"]["dtype"], batch_size=batch_size,
                       device="cuda" if torch.cuda.is_available() else "cpu")


def run_lm_eval(lm, task: str, chat: bool, cfg) -> dict:
    from lm_eval import simple_evaluate

    kw = dict(model=lm, tasks=[task], num_fewshot=0, log_samples=True, apply_chat_template=chat,
              random_seed=cfg["seed"], numpy_random_seed=cfg["seed"], torch_random_seed=cfg["seed"],
              fewshot_random_seed=cfg["seed"])
    if task.startswith("gsm8k"):
        kw["gen_kwargs"] = f"max_gen_toks={cfg['eval']['max_gen_toks']},do_sample=False,temperature=0.0"
    return simple_evaluate(**kw)


def gsm8k_rows(samples, tok, max_gen: int) -> list[dict]:
    """One row per question: generation, extracted answers per filter, correctness, truncation flag."""
    by_doc: dict[int, dict] = {}
    for s in samples:
        d = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"],
                                            "target": s["target"], "response": s["resps"][0][0]})
        d[f"extracted_{s['filter']}"] = s["filtered_resps"][0]
        d[f"correct_{s['filter']}"] = float(s.get("exact_match", 0.0))
    for d in by_doc.values():
        n = len(tok(d["response"], add_special_tokens=False)["input_ids"])
        d["n_gen_tokens"] = n
        d["truncated"] = n >= max_gen - 1          # hit the 1,024-token cap without a stop sequence
        d["unextractable"] = d.get("extracted_flexible-extract", "[invalid]") == "[invalid]"
    return sorted(by_doc.values(), key=lambda r: r["doc_id"])


def ll_rows(samples) -> list[dict]:
    rows = []
    for s in samples:
        rows.append({"doc_id": s["doc_id"], "target": s["target"],
                     "loglik": [r[0] for r in s["filtered_resps"]] if s.get("filtered_resps") else None,
                     "acc": s.get("acc"), "acc_norm": s.get("acc_norm")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--condition", required=True, help="baseline | full_ft | l2sp | lora4 | lora16 | oplora16")
    ap.add_argument("--update", type=int, default=1250)
    ap.add_argument("--tasks", default="gsm8k,hellaswag,boolq,arc_easy")
    ap.add_argument("--limit", type=int, default=None, help="debug only – never use for reported numbers")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cond_name(args.condition, args.update)
    res = ensure_dir(Path(cfg["paths"]["results"]) / "eval" / name)

    path, meta = build_eval_model_dir(cfg, args.condition, args.update, device)
    save_json(meta, res / "build.json")
    tok = load_tokenizer(cfg["model"])
    scores = {}
    if (res / "scores.json").exists():
        import json
        scores = json.load(open(res / "scores.json"))
    for key in args.tasks.split(","):
        if key in scores:
            print("skip", key)
            continue
        tcfg = cfg["eval"]["tasks"][key]
        bs = cfg["eval"]["gen_batch_size"] if key == "gsm8k" else cfg["eval"]["batch_size"]
        lm = make_lm(path, cfg, bs)
        if args.limit:
            from lm_eval import simple_evaluate
            out = simple_evaluate(model=lm, tasks=[tcfg["task"]], num_fewshot=0, log_samples=True,
                                  apply_chat_template=tcfg["chat"], limit=args.limit,
                                  gen_kwargs=f"max_gen_toks={cfg['eval']['max_gen_toks']},do_sample=False")
        else:
            out = run_lm_eval(lm, tcfg["task"], tcfg["chat"], cfg)
        r = out["results"][tcfg["task"]]
        samples = out["samples"][tcfg["task"]]
        if key == "gsm8k":
            rows = gsm8k_rows(samples, tok, cfg["eval"]["max_gen_toks"])
            scores[key] = {"primary": r.get("exact_match,flexible-extract"),
                           "strict": r.get("exact_match,strict-match"),
                           "truncation_rate": sum(x["truncated"] for x in rows) / len(rows),
                           "unextractable_rate": sum(x["unextractable"] for x in rows) / len(rows),
                           "mean_gen_tokens": sum(x["n_gen_tokens"] for x in rows) / len(rows), "n": len(rows)}
        else:
            rows = ll_rows(samples)
            prim = "acc" if key == "boolq" else "acc_norm"
            scores[key] = {"primary": r.get(f"{prim},none"), "acc": r.get("acc,none"),
                           "acc_norm": r.get("acc_norm,none"), "n": len(rows)}
        scores[key]["lm_eval_task"] = tcfg["task"]
        write_jsonl(rows, res / f"samples_{key}.jsonl")
        save_json(scores, res / "scores.json")
        del lm, out
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(key, scores[key])
    try:
        import lm_eval
        scores["_lm_eval_version"] = lm_eval.__version__
    except Exception:
        pass
    save_json(scores, res / "scores.json")
    # merged FP16 models in outputs/task1/merged/ are reused by qualitative.py


if __name__ == "__main__":
    main()
