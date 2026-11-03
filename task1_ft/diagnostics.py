"""Task 1 Step 6: weight-space and activation diagnostics on down_proj (layers 0, 13, 27).

    python -m task1_ft.diagnostics --config configs/task1.yaml probes      # BEFORE inspecting any model results
    python -m task1_ft.diagnostics --config configs/task1.yaml cache       # cache W0 z inputs from the starting model
    python -m task1_ft.diagnostics --config configs/task1.yaml analyze     # all checkpoints that exist

rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2  (= ||U_k^T dW||_F^2 / ||dW||_F^2, U_k orthonormal)
eta   = ||dW||_F / ||W0||_F
R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8), z = cached down_proj input at the last prompt token.
All in FP32. Full FT: dW = W_trained - W0. LoRA: (alpha/r) B A. OPLoRA: (alpha/r) B~ A~ (projected factors).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.utils import ensure_dir, load_cfg, load_json, save_json
from task1_ft.data import load_tokenizer

CONDITIONS = [("full_ft", 625), ("full_ft", 1250), ("l2sp", 1250), ("lora4", 1250), ("lora16", 1250),
              ("oplora16", 1250)]


# ------------------------------------------------------------------ probe set
def hellaswag_preprocess(text: str) -> str:
    """lm-eval's HellaSwag preprocessing."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def build_probes(cfg) -> dict:
    from datasets import load_dataset

    rng = np.random.default_rng(cfg["seed"])
    gsm = load_dataset(cfg["data"]["gsm8k_dataset"], "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    gi = sorted(rng.choice(len(gsm), cfg["diagnostics"]["n_probe_math"], replace=False).tolist())
    hi = sorted(rng.choice(len(hs), cfg["diagnostics"]["n_probe_other"], replace=False).tolist())
    probes = []
    for i in gi:
        probes.append({"kind": "math", "dataset": "gsm8k_test", "idx": i,
                       "user_text": f"Q: {gsm[i]['question']}\nA:", "chat": True})  # gsm8k_cot zero-shot format
    for i in hi:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        probes.append({"kind": "non_math", "dataset": "hellaswag_validation", "idx": i,
                       "text": hellaswag_preprocess(d["activity_label"] + ": " + ctx), "chat": False})
    return {"seed": cfg["seed"], "gsm8k_test_ids": gi, "hellaswag_val_ids": hi, "probes": probes}


def probe_text(tok, p) -> str:
    if p["chat"]:
        return tok.apply_chat_template([{"role": "user", "content": p["user_text"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    return p["text"]


@torch.no_grad()
def cache_inputs(cfg) -> None:
    """Run the *starting* model in FP32 and cache down_proj inputs at the last prompt token."""
    from transformers import AutoModelForCausalLM

    dc = cfg["diagnostics"]
    sd = Path(cfg["data"]["split_dir"])
    probes = load_json(sd / "probe_set.json")["probes"]
    tok = load_tokenizer(cfg["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(device).eval()
    store = {l: [] for l in dc["layers"]}
    hooks = []
    for l in dc["layers"]:
        mod = getattr(model.model.layers[l].mlp, dc["module"])
        hooks.append(mod.register_forward_pre_hook(lambda m, inp, l=l: store[l].append(inp[0][0, -1].float().cpu())))
    for p in probes:
        ids = tok(probe_text(tok, p), return_tensors="pt", add_special_tokens=False).to(device)  # batch 1: no padding
        model(**ids)
    for h in hooks:
        h.remove()
    out = Path(cfg["paths"]["outputs"]) / "diagnostics"
    ensure_dir(out)
    torch.save({"z": {l: torch.stack(v) for l, v in store.items()}, "kinds": [p["kind"] for p in probes]},
               out / "probe_inputs.pt")
    print("cached", {l: tuple(torch.stack(v).shape) for l, v in store.items()})


# ------------------------------------------------------------------ weight-space
def base_weights(cfg) -> dict:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    mod = cfg["diagnostics"]["module"]
    return {l: getattr(m.model.layers[l].mlp, mod).weight.detach().clone() for l in cfg["diagnostics"]["layers"]}


def delta_weights(cfg, method: str, update: int, W0: dict) -> dict | None:
    wdir = Path(cfg["paths"]["outputs"]) / method / f"weights_{update:04d}"
    mod = cfg["diagnostics"]["module"]
    layers = cfg["diagnostics"]["layers"]
    if cfg["methods"][method]["kind"] == "full":
        f = wdir / "model_fp32.pt"
        if not f.exists():
            return None
        sd = torch.load(f, map_location="cpu", mmap=True)
        return {l: sd[f"model.layers.{l}.mlp.{mod}.weight"].float() - W0[l] for l in layers}
    f = wdir / "effective_factors.pt"
    if not f.exists():
        return None
    fac = torch.load(f, map_location="cpu")
    out = {}
    for l in layers:
        key = f"model.layers.{l}.mlp.{mod}"
        out[l] = float(fac[f"{key}.scaling"]) * (fac[f"{key}.B_eff"].float() @ fac[f"{key}.A_eff"].float())
    return out


def analyze(cfg) -> None:
    dc = cfg["diagnostics"]
    k = dc["k"]
    res = ensure_dir(Path(cfg["paths"]["results"]) / "diagnostics")
    W0 = base_weights(cfg)
    svd = {}
    for l, W in W0.items():
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        svd[l] = (U[:, :k], Vh[:k].T)
    cache = torch.load(Path(cfg["paths"]["outputs"]) / "diagnostics" / "probe_inputs.pt")
    kinds = np.array(cache["kinds"])
    ws_rows, act_rows, per_probe = [], [], []
    for method, upd in CONDITIONS:
        dW = delta_weights(cfg, method, upd, W0)
        if dW is None:
            print("missing", method, upd)
            continue
        name = f"{method}_{upd}"
        for l in dc["layers"]:
            D, W = dW[l], W0[l]
            Uk, Vk = svd[l]
            fro2 = float((D * D).sum())
            rho = float((Uk.T @ D).pow(2).sum()) / fro2 if fro2 > 0 else float("nan")
            row = {"condition": name, "layer": l, "rho_k": rho, "eta": fro2 ** 0.5 / float(W.norm()),
                   "dW_fro": fro2 ** 0.5, "rho_k_input_side": float((D @ Vk).pow(2).sum()) / fro2 if fro2 > 0 else float("nan")}
            ws_rows.append(row)
            z = cache["z"][l]                                  # n x d_in
            R = (z @ D.T).norm(dim=1) / ((z @ W.T).norm(dim=1) + 1e-8)
            for i, r in enumerate(R.tolist()):
                per_probe.append({"condition": name, "layer": l, "probe": i, "kind": kinds[i], "R": r})
            for kind in ("math", "non_math"):
                Rk = R[torch.from_numpy(kinds == kind)]
                act_rows.append({"condition": name, "layer": l, "kind": kind, "R_mean": float(Rk.mean()),
                                 "R_median": float(Rk.median()), "R_std": float(Rk.std())})
    ws, act = pd.DataFrame(ws_rows), pd.DataFrame(act_rows)
    ws.to_csv(res / "weight_space.csv", index=False)
    act.to_csv(res / "activation.csv", index=False)
    pd.DataFrame(per_probe).to_csv(res / "activation_per_probe.csv", index=False)
    if len(act):
        piv = act.pivot_table(index=["condition", "layer"], columns="kind", values="R_mean").reset_index()
        piv["math_over_non_math"] = piv["math"] / piv["non_math"]
        piv.to_csv(res / "activation_ratio.csv", index=False)
    print(ws.round(4).to_string())
    print(act.round(4).to_string())


@torch.no_grad()
def verify_oplora(cfg) -> None:
    """Numerical check of OPLoRA constraints over *all* adapted modules: U_k^T B~ = 0 and A~ V_k = 0."""
    from transformers import AutoModelForCausalLM

    wdir = Path(cfg["paths"]["outputs"]) / "oplora16" / "weights_1250"
    fac = torch.load(wdir / "effective_factors.pt", map_location="cpu")
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    k = cfg["methods"]["oplora16"]["k"]
    worst = {"rel_UkT_dW": 0.0, "rel_dW_Vk": 0.0}
    rows = []
    for name, mod in base.named_modules():
        key = f"{name}.B_eff"
        if key not in fac:
            continue
        U, S, Vh = torch.linalg.svd(mod.weight.float(), full_matrices=False)
        Uk, Vk = U[:, :k], Vh[:k].T
        D = float(fac[f"{name}.scaling"]) * fac[key] @ fac[f"{name}.A_eff"]
        n = float(D.norm()) + 1e-30
        a, b = float((Uk.T @ D).norm()) / n, float((D @ Vk).norm()) / n
        rows.append({"module": name, "rel_UkT_dW": a, "rel_dW_Vk": b})
        worst = {"rel_UkT_dW": max(worst["rel_UkT_dW"], a), "rel_dW_Vk": max(worst["rel_dW_Vk"], b)}
    res = ensure_dir(Path(cfg["paths"]["results"]) / "diagnostics")
    pd.DataFrame(rows).to_csv(res / "oplora_constraints.csv", index=False)
    save_json(worst, res / "oplora_constraints_worst.json")
    print(worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("step", choices=["probes", "cache", "analyze", "verify_oplora"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.step == "probes":
        p = Path(cfg["data"]["split_dir"]) / "probe_set.json"
        if p.exists():
            print("probe set already fixed:", p)
        else:
            save_json(build_probes(cfg), p)
            print("saved", p)
    elif args.step == "cache":
        cache_inputs(cfg)
    elif args.step == "analyze":
        analyze(cfg)
    else:
        verify_oplora(cfg)


if __name__ == "__main__":
    main()
