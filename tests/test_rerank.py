import pytest

from lra.retrieval.hybrid import FusedHit
from lra.retrieval.keyword import Hit
from lra.retrieval.rerank import rerank, sigmoid


def fused(num: str, rank: int, text: str) -> FusedHit:
    return FusedHit(
        Hit(rank, f"ID-{num}", num, "VIGUEUR", "2019-04-01", None, "P", text, 0.0), {"vector": rank}
    )


class WordScorer:
    """Score = nombre de mots de la question présents dans le document, moins 2."""

    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append(documents)
        words = set(query.lower().split())
        return [float(sum(w in d.lower() for w in words) - 2) for d in documents]


def test_rerank_reorders_and_keeps_hybrid_rank():
    cands = [fused("A", 1, "rien"), fused("B", 2, "durée accord-cadre"), fused("C", 3, "durée")]
    scorer = WordScorer()
    out = rerank("durée accord-cadre", cands, scorer, k=2)
    assert [r.fused.hit.num for r in out] == ["B", "C"]
    assert [r.fused.hit.rank for r in out] == [1, 2]
    assert [r.hybrid_rank for r in out] == [2, 3]
    assert out[0].rerank_score == pytest.approx(sigmoid(0.0))
    # Le reranker voit le contexte de l'article, pas seulement son texte.
    assert scorer.calls[0][0] == "P\nArticle A\nrien"


def test_rerank_empty():
    assert rerank("q", [], WordScorer(), k=5) == []


def test_sigmoid_is_stable_for_large_values():
    assert sigmoid(0) == 0.5
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(-1000) == pytest.approx(0.0)
