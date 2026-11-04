"""Adapters (LoRA, OPLoRA), L2-SP penalty and model construction for the five Task 1 variants.

LoRA:    W = W0 + (alpha/r) B A,     A in R^{r x d_in} (Kaiming-uniform, as in PEFT), B in R^{d_out x r} = 0.
OPLoRA:  W0 = U S V^T (FP32 SVD, computed once). With the top-k bases U_k, V_k frozen,
         B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T,  dW = (alpha/r) B~ A~.
         Effective factors are formed on the fly (r x k products); no d x d projector is materialised.
The adapter path runs in FP32 on top of a frozen FP16 backbone.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, d_in, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # PEFT default: random A, zero B
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    # effective factors (identity for plain LoRA)
    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lora_B, self.lora_A

    def delta_weight(self) -> torch.Tensor:
        B, A = self.factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        B, A = self.factors()
        h = self.dropout(x).to(A.dtype)
        # run the low-rank path in FP32 even under autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            d = F.linear(F.linear(h.float(), A.float()), B.float()) * self.scaling
        return y + d.to(y.dtype)


class OPLoRALinear(LoRALinear):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, k: int):
        super().__init__(base, r, alpha, dropout)
        W0 = base.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        self.k = k
        self.register_buffer("U_k", U[:, :k].contiguous(), persistent=True)        # d_out x k (FP32)
        self.register_buffer("V_k", Vh[:k, :].T.contiguous(), persistent=True)     # d_in  x k (FP32)

    def factors(self):
        B, A = self.lora_B, self.lora_A
        B_t = B - self.U_k @ (self.U_k.T @ B)
        A_t = A - (A @ self.V_k) @ self.V_k.T
        return B_t, A_t


def _get_parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_adapters(model: nn.Module, kind: str, r: int, alpha: float, dropout: float, targets: list[str],
                    k: int | None = None) -> dict:
    """Replace every target nn.Linear in place. Order of replacement (module order) fixes the RNG stream,
    so LoRA-16 and OPLoRA-16 receive identical initial A, B under the same seed."""
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    t0 = time.perf_counter()
    for n in names:
        parent, attr = _get_parent(model, n)
        base = getattr(parent, attr)
        mod = OPLoRALinear(base, r, alpha, dropout, k) if kind == "oplora" else LoRALinear(base, r, alpha, dropout)
        setattr(parent, attr, mod)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    info = {"n_adapted_modules": len(names), "inject_seconds": time.perf_counter() - t0}
    if kind == "oplora":
        nbytes = sum(b.numel() * b.element_size() for m in model.modules() if isinstance(m, OPLoRALinear)
                     for b in (m.U_k, m.V_k))
        info.update({"svd_preprocess_seconds": info["inject_seconds"], "basis_storage_mb": nbytes / 1024**2})
    return info


def adapter_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def adapter_state_dict(model: nn.Module, include_bases: bool = False) -> dict:
    sd = {}
    for n, m in adapter_modules(model):
        sd[f"{n}.lora_A"] = m.lora_A.detach().cpu()
        sd[f"{n}.lora_B"] = m.lora_B.detach().cpu()
        if include_bases and isinstance(m, OPLoRALinear):
            sd[f"{n}.U_k"] = m.U_k.cpu()
            sd[f"{n}.V_k"] = m.V_k.cpu()
    return sd


def export_effective_factors(model: nn.Module) -> dict:
    """Projected factors (B~, A~) for OPLoRA, raw factors for LoRA – what inference / merging uses."""
    out = {}
    for n, m in adapter_modules(model):
        with torch.no_grad():
            B, A = m.factors()
        out[f"{n}.B_eff"] = B.detach().float().cpu()
        out[f"{n}.A_eff"] = A.detach().float().cpu()
        out[f"{n}.scaling"] = torch.tensor(m.scaling)
    return out


@torch.no_grad()
def merge_effective_factors(model: nn.Module, factors: dict) -> int:
    """Merge exported effective factors into a fresh (plain) backbone: W <- W0 + s * B_eff A_eff."""
    n_merged = 0
    for name, mod in model.named_modules():
        key = f"{name}.B_eff"
        if isinstance(mod, nn.Linear) and key in factors:
            B, A, s = factors[key], factors[f"{name}.A_eff"], float(factors[f"{name}.scaling"])
            dW = (s * (B.to(mod.weight.device) @ A.to(mod.weight.device))).to(torch.float32)
            mod.weight.data = (mod.weight.data.float() + dW).to(mod.weight.dtype)
            n_merged += 1
    return n_merged


def load_adapter_state(model: nn.Module, sd: dict) -> None:
    for n, m in adapter_modules(model):
        m.lora_A.data.copy_(sd[f"{n}.lora_A"])
        m.lora_B.data.copy_(sd[f"{n}.lora_B"])


# ---------------------------------------------------------------- L2-SP
class L2SPReference:
    """Frozen copy of theta_0 for every unique trainable parameter (tied weights counted once because
    `named_parameters()` de-duplicates shared tensors). Stored in the checkpoint's native dtype (bf16 for
    Qwen3), which is an exact copy of the starting weights."""

    def __init__(self, model: nn.Module, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        t0 = time.perf_counter()
        self.ref = {n: p.detach().to(device=device, dtype=dtype).clone()
                    for n, p in model.named_parameters() if p.requires_grad}
        if device == "cpu" and torch.cuda.is_available():   # pinned host memory -> fast chunk transfers
            self.ref = {n: t.pin_memory() for n, t in self.ref.items()}
        # exactness check: theta0 was loaded from bf16, so casting back must be lossless
        self.max_roundtrip_err = max(float((p.detach().float() - self.ref[n].float().to(p.device)).abs().max())
                                     for n, p in model.named_parameters() if n in self.ref)
        self.dtype = str(dtype)
        if self.max_roundtrip_err > 0:  # storage dtype is lossy for these weights -> keep an exact FP32 copy
            self.ref = {n: p.detach().to(device=device, dtype=torch.float32).clone()
                        for n, p in model.named_parameters() if n in self.ref}
            self.dtype = "torch.float32 (fallback)"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - t0
        self.storage_mb = sum(t.numel() * t.element_size() for t in self.ref.values()) / 1024**2
        self.n_params = sum(t.numel() for t in self.ref.values())
        self.penalty_seconds = 0.0

    def omega(self, model: nn.Module) -> torch.Tensor:
        """Omega = 0.5 * sum_j (theta_j - theta0_j)^2, computed in FP32 (summed, not averaged)."""
        t0 = time.perf_counter()
        tot = None
        for n, p in model.named_parameters():
            if n not in self.ref:
                continue
            d = p.float() - self.ref[n].to(device=p.device, dtype=torch.float32)
            s = (d * d).sum()
            tot = s if tot is None else tot + s
        self.penalty_seconds += time.perf_counter() - t0
        return 0.5 * tot

    @torch.no_grad()
    def add_penalty_grad(self, model: nn.Module, coef: float, chunk_rows: int = 8192) -> float:
        """Add coef * (theta - theta0) to every parameter's .grad in place and return Omega (FP32).

        This is exactly d(lambda * Omega)/d theta (with coef = lambda * loss_scale), but avoids building an
        autograd graph over all ~600M parameters, which would store a full FP32 copy of (theta - theta0) and does
        not fit next to FP32 weights + AdamW states on a 16 GB GPU. Large tensors are processed in row chunks so
        the temporary is small."""
        t0 = time.perf_counter()
        tot = 0.0
        for n, p in model.named_parameters():
            if n not in self.ref:
                continue
            ref = self.ref[n]
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            P, G = p.data, p.grad
            if P.ndim == 0:
                P, G, ref = P.view(1), G.view(1), ref.view(1)
            for s in range(0, P.shape[0], chunk_rows):
                d = P[s:s + chunk_rows].float() - ref[s:s + chunk_rows].to(device=P.device, non_blocking=True).float()
                tot += float((d * d).sum())
                G[s:s + chunk_rows].add_(d.to(G.dtype), alpha=coef)
        self.penalty_seconds += time.perf_counter() - t0
        return 0.5 * tot

    @torch.no_grad()
    def displacement(self, model: nn.Module) -> float:
        return float(torch.sqrt(2 * self.omega(model)))


@torch.no_grad()
def param_displacement(model: nn.Module, ref_model: nn.Module) -> float:
    """||theta - theta0||_2 over unique parameters."""
    ref = dict(ref_model.named_parameters())
    tot = 0.0
    for n, p in model.named_parameters():
        d = p.detach().float() - ref[n].detach().float().to(p.device)
        tot += float((d * d).sum())
    return math.sqrt(tot)


# ---------------------------------------------------------------- model construction
def load_base(model_name: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    return model.to(device)


def build_model(cfg: dict, method: str, device: str):
    """Return (model, info). Full FT: FP32 master weights. Adapters: FP16 frozen backbone + FP32 adapters."""
    m = cfg["methods"][method]
    info = {"method": method, **m}
    if m["kind"] == "full":
        model = load_base(cfg["model"], torch.float32, device)
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        model = load_base(cfg["model"], torch.float16, device)
        for p in model.parameters():
            p.requires_grad_(False)
        lc = cfg["lora_common"]
        info.update(inject_adapters(model, m["kind"], m["r"], m["alpha"], lc["dropout"], lc["targets"], m.get("k")))
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_params"] = sum(p.numel() for p in model.parameters())
    return model, info


def param_groups(model: nn.Module, weight_decay: float):
    """AdamW groups: no decay for biases and parameters with < 2 dims (norm weights)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
