"""Quality scores and answer-selection rules over a candidate set (Task 3, Step 2).

All rules see only the candidates in the current prefix (no gold answers, no unrevealed rollouts).
Ties – both score ties and vote ties – are resolved by the earliest supporting rollout in the current order.
RWC uses average ranks for tied scores.

Scores per rollout i (T_i scored tokens):
  L_i = mean log p(y_t)                 (sampled-token support)
  K_i = mean KL(U || p_t)               (self-certainty)
  V_i = mean A_img / (A_img + A_prompt + A_gen)   (visual engagement, MathVista)
  C_i = mean clip01((k_t - Q05(G)) / max(Q95(G) - Q05(G), 1e-8)),  Q over all tokens of the current set G
  Z_i = V_i * C_i (VisConf)      Z_tok_i = mean_t c_t v_t (tokenwise fusion ablation)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

NO_ANSWER = -1


@dataclass
class Question:
    qid: str
    groups: np.ndarray            # [R] answer-group id per rollout (-1 = NO_ANSWER)
    correct: np.ndarray           # [R] bool – rollout's group is correct
    n_tokens: np.ndarray          # [R] retained generated tokens (cost)
    L: np.ndarray
    K: np.ndarray
    V: np.ndarray | None = None
    kl_tok: list = field(default_factory=list)   # R arrays
    v_tok: list | None = None


# ------------------------------------------------------------- primitives
def earliest_argmax(values: np.ndarray, tol: float = 0.0) -> int:
    """Index (into the candidate order) of the maximum; ties -> earliest position."""
    m = np.max(values)
    return int(np.flatnonzero(values >= m - tol)[0])


def vote_winner(groups: np.ndarray, weights: np.ndarray) -> tuple[int, dict]:
    """Group with the largest total weight; ties -> group whose earliest supporting rollout comes first."""
    tot, first = {}, {}
    for pos, (g, w) in enumerate(zip(groups.tolist(), weights.tolist())):
        tot[g] = tot.get(g, 0.0) + w
        first.setdefault(g, pos)
    best = max(tot.values())
    cands = [g for g, v in tot.items() if np.isclose(v, best, rtol=0, atol=1e-12)]
    return min(cands, key=lambda g: first[g]), tot


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = rankdata(-scores, method="average")   # rank 1 = best
    return (n - r) / (n - 1)


def rwc_margin(tot: dict) -> float:
    v = sorted(tot.values(), reverse=True)
    s = sum(v)
    if s <= 0:
        return 0.0
    return (v[0] - (v[1] if len(v) > 1 else 0.0)) / s


def visconf_scores(q: Question, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Z, Z_tok) for the candidates, calibrated on the pooled KL tokens of *this* candidate set."""
    pool = np.concatenate([q.kl_tok[i] for i in cand]) if len(cand) else np.zeros(0)
    if pool.size == 0:
        return np.zeros(len(cand)), np.zeros(len(cand))
    q05, q95 = np.percentile(pool, [5, 95])          # linear interpolation (numpy default)
    den = max(q95 - q05, 1e-8)
    Z, Zt = np.zeros(len(cand)), np.zeros(len(cand))
    for j, i in enumerate(cand):
        k, v = q.kl_tok[i], q.v_tok[i]
        if len(k) == 0:
            continue
        c = np.clip((k - q05) / den, 0.0, 1.0)
        Z[j] = q.V[i] * c.mean()
        Zt[j] = float(np.mean(c * v))
    return Z, Zt


# ------------------------------------------------------------- selection
def method_names(vlm: bool) -> list[str]:
    base = ["firstans", "sc", "max_L", "max_K", "rwc_K", "shortest"]
    if vlm:
        base += ["max_V", "rwc_V", "max_Z", "rwc_Z", "max_Ztok", "rwc_Ztok"]
    return base


def score_table(q: Question, cand: np.ndarray, vlm: bool) -> dict:
    s = {"L": q.L[cand], "K": q.K[cand], "neg_len": -q.n_tokens[cand].astype(float)}
    if vlm:
        s["V"] = q.V[cand]
        s["Z"], s["Ztok"] = visconf_scores(q, cand)
    return s


def select_all(q: Question, cand: np.ndarray, vlm: bool) -> dict:
    """Return {method: (selected_group, cost, extra)} for one candidate prefix."""
    g = q.groups[cand]
    cost_all = int(q.n_tokens[cand].sum())
    out = {"firstans": (int(g[0]), int(q.n_tokens[cand[0]]), {})}
    sc_g, sc_tot = vote_winner(g, np.ones(len(cand)))
    out["sc"] = (sc_g, cost_all, {"votes": sc_tot})
    S = score_table(q, cand, vlm)
    for name, key in [("max_L", "L"), ("max_K", "K"), ("shortest", "neg_len")] + (
            [("max_V", "V"), ("max_Z", "Z"), ("max_Ztok", "Ztok")] if vlm else []):
        out[name] = (int(g[earliest_argmax(S[key])]), cost_all, {})
    for name, key in [("rwc_K", "K")] + ([("rwc_V", "V"), ("rwc_Z", "Z"), ("rwc_Ztok", "Ztok")] if vlm else []):
        gw, tot = vote_winner(g, rwc_weights(S[key]))
        out[name] = (gw, cost_all, {"margin": rwc_margin(tot)})
    return out


def is_correct(q: Question, group: int) -> bool:
    if group == NO_ANSWER:
        return False
    idx = np.flatnonzero(q.groups == group)
    return bool(q.correct[idx[0]])


def make_orders(qids: list[str], R: int, n_orders: int, seed: int) -> dict:
    """Ten seeded random permutations of the 32 rollout IDs per question (one generator, qids sorted)."""
    rng = np.random.default_rng(seed)
    return {qid: np.stack([rng.permutation(R) for _ in range(n_orders)]) for qid in sorted(qids)}
