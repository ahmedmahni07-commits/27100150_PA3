"""CPU checks for Task 3 selection rules.   python -m tests.test_task3"""
import numpy as np

from task3_ttc.grading import NO_ANSWER, group_answers, mathvista_gold, mathvista_normalize
from task3_ttc.live import last_boxed
from task3_ttc.selection import Question, rwc_margin, rwc_weights, select_all, visconf_scores, vote_winner


def test_vote_ties_earliest():
    g, _ = vote_winner(np.array([2, 1, 1, 2]), np.ones(4))
    assert g == 2                                     # 2-2 tie -> group of rollout at position 0
    g, tot = vote_winner(np.array([NO_ANSWER, NO_ANSWER, 0]), np.ones(3))
    assert g == NO_ANSWER and tot[NO_ANSWER] == 2     # NO_ANSWER still votes


def test_rwc_average_ranks():
    w = rwc_weights(np.array([3.0, 1.0, 3.0, 0.0]))
    # ranks: 1.5, 3, 1.5, 4 -> w = (4 - r) / 3
    assert np.allclose(w, [2.5 / 3, 1 / 3, 2.5 / 3, 0])
    assert rwc_weights(np.array([5.0]))[0] == 1
    assert np.isclose(rwc_margin({0: 3.0, 1: 1.0}), 0.5) and rwc_margin({0: 2.0}) == 1.0


def test_n1_all_rules_agree_and_visconf_identity():
    rng = np.random.default_rng(0)
    R = 6
    kl = [rng.random(int(rng.integers(3, 9))) * 5 for _ in range(R)]
    v = [rng.random(len(k)) for k in kl]
    q = Question("q", groups=np.array([0, 1, 1, NO_ANSWER, 2, 1]), correct=np.array([0, 1, 1, 0, 0, 1], bool),
                 n_tokens=np.array([len(k) for k in kl]), L=rng.random(R), K=np.array([k.mean() for k in kl]),
                 V=np.array([x.mean() for x in v]), kl_tok=kl, v_tok=v)
    sel = select_all(q, np.array([4]), vlm=True)
    assert {s[0] for s in sel.values()} == {2}
    cand = np.arange(R)
    Z, Zt = visconf_scores(q, cand)
    pool = np.concatenate(kl)
    lo, hi = np.percentile(pool, [5, 95])
    for i in cand:
        c = np.clip((kl[i] - lo) / (hi - lo), 0, 1)
        cov = np.mean(c * v[i]) - c.mean() * v[i].mean()
        assert np.isclose(Zt[i] - Z[i], cov)          # Z_tok - Z = Cov_t(c, v)


def test_grading_rules():
    meta = {"question_type": "multi_choice", "answer_type": "text", "choices": ["red", "blue"], "answer": "blue"}
    assert mathvista_gold(meta) == "B"
    assert mathvista_normalize("(B) blue", meta) == "B" and mathvista_normalize("Blue", meta) == "B"
    fmeta = {"question_type": "free_form", "answer_type": "float", "precision": 1, "answer": "1.5"}
    assert mathvista_normalize("1.50", fmeta) == "1.5" and mathvista_normalize("abc", fmeta) is None
    assert group_answers(["a", None, "b", "a"], lambda x, y: x == y) == [0, NO_ANSWER, 1, 0]
    assert last_boxed(r"x \boxed{1} y \boxed{\frac{1}{2}}") == r"\frac{1}{2}" and last_boxed(r"\boxed{a") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
