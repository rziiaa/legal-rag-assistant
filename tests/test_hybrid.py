import pytest

from lra.retrieval.hybrid import rrf
from lra.retrieval.keyword import Hit


def hit(num: str, rank: int) -> Hit:
    return Hit(rank, f"ID-{num}", num, "VIGUEUR", "2019-04-01", None, "", "", 0.0)


def test_article_found_by_both_methods_comes_first():
    lists = {
        "keyword": [hit("A", 1), hit("B", 2), hit("C", 3)],
        "vector": [hit("C", 1), hit("D", 2), hit("B", 3)],
    }
    fused = rrf(lists, k=10, rrf_k=60)
    assert [f.hit.num for f in fused] == ["C", "B", "A", "D"]
    assert fused[0].ranks == {"keyword": 3, "vector": 1}
    assert fused[0].hit.score == pytest.approx(1 / 63 + 1 / 61)
    assert [f.hit.rank for f in fused] == [1, 2, 3, 4]


def test_k_limits_results_and_ties_are_stable():
    lists = {"keyword": [hit("B", 1)], "vector": [hit("A", 1)]}
    fused = rrf(lists, k=1)
    assert [f.hit.num for f in fused] == ["A"]  # égalité : ordre par numéro


def test_empty_lists():
    assert rrf({"keyword": [], "vector": []}, k=5) == []
