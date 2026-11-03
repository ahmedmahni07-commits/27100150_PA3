"""CPU checks for Task 1 building blocks on a tiny random Qwen3 (no downloads).

    python -m tests.test_task1
"""
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from task1_ft.data import IGNORE, collate, n_target_tokens
from task1_ft.methods import (L2SPReference, LoRALinear, OPLoRALinear, export_effective_factors, inject_adapters,
                              merge_effective_factors, param_groups)
from task1_ft.train import response_ce_sum

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, tie_word_embeddings=True, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).eval()


def test_masked_loss_matches_reference():
    m = tiny()
    ids = [torch.randint(0, 97, (n,)) for n in (11, 7)]
    labs = [x.clone() for x in ids]
    labs[0][:5] = IGNORE
    labs[1][:3] = IGNORE
    b = collate(ids, labs, pad_id=0)
    ours = response_ce_sum(m, b, "cpu") / n_target_tokens(b["labels"])
    logits = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 97), b["labels"][:, 1:].reshape(-1), ignore_index=IGNORE)
    assert torch.allclose(ours, ref, atol=1e-5), (ours, ref)


def test_lora_identity_at_init_and_merge():
    m = tiny()
    x = torch.randint(0, 97, (2, 9))
    base = m(x).logits
    torch.manual_seed(1)
    info = inject_adapters(m, "lora", 4, 8, 0.0, TARGETS)
    assert info["n_adapted_modules"] == 14
    assert torch.allclose(m(x).logits, base, atol=1e-5)          # B = 0
    for _, mod in m.named_modules():
        if isinstance(mod, LoRALinear):
            mod.lora_B.data.normal_(0, 0.1)
    unmerged = m(x).logits
    fresh = tiny()
    merge_effective_factors(fresh, export_effective_factors(m))
    assert torch.allclose(fresh(x).logits, unmerged, atol=1e-4)


def test_oplora_constraints_and_shared_init():
    m1, m2 = tiny(), tiny()
    torch.manual_seed(5)
    inject_adapters(m1, "lora", 16, 32, 0.0, TARGETS)
    torch.manual_seed(5)
    inject_adapters(m2, "oplora", 16, 32, 0.0, TARGETS, k=8)
    for (n1, a), (n2, b) in zip(m1.named_modules(), m2.named_modules()):
        if isinstance(a, LoRALinear):
            assert torch.equal(a.lora_A, b.lora_A), n1               # same unprojected initial factors
    for _, mod in m2.named_modules():
        if isinstance(mod, OPLoRALinear):
            mod.lora_B.data.normal_(0, 0.1)
            dW = mod.delta_weight()
            assert (mod.U_k.T @ dW).norm() / dW.norm() < 1e-5
            assert (dW @ mod.V_k).norm() / dW.norm() < 1e-5
            # gradients flow to raw factors through the projection
    x = torch.randint(0, 97, (2, 9))
    m2(x).logits.sum().backward()
    assert all(p.grad is not None for n, p in m2.named_parameters() if "lora_" in n)
    fresh = tiny()
    merge_effective_factors(fresh, export_effective_factors(m2))
    assert torch.allclose(fresh(x).logits, m2(x).logits, atol=1e-4)


def test_l2sp_omega_and_tied_params():
    m = tiny()
    ref = L2SPReference(m, device="cpu", dtype=torch.float32)
    n_unique = sum(p.numel() for p in m.parameters())
    assert ref.n_params == n_unique                                   # tied embedding counted once
    assert float(ref.omega(m)) == 0.0
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01)
    assert abs(float(ref.omega(m)) - 0.5 * n_unique * 1e-4) / (0.5 * n_unique * 1e-4) < 1e-3
    groups = param_groups(m, 0.01)
    assert all(p.ndim >= 2 for p in groups[0]["params"]) and all(p.ndim < 2 for p in groups[1]["params"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
