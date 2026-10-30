import math

from ragstack.application.evaluation.metrics import leak_rate, mrr, ndcg_at_k, recall_at_k, refusal_correct


def test_rank_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert mrr(["x", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["b"], {"b"}, 5) == 1.0
    assert 0 < ndcg_at_k(["x", "b"], {"b"}, 5) < 1
    assert math.isnan(recall_at_k(["a"], set(), 3))


def test_safety_metrics():
    assert leak_rate(["d1", "d2", "d1"], {"d1"}) == 2 / 3
    assert leak_rate([], {"d1"}) == 0.0
    assert refusal_correct(True, True) == 1.0 and refusal_correct(False, True) == 0.0
