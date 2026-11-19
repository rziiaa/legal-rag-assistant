"""Reranking : un second modèle relit chaque paire (question, article) et réordonne les candidats.

Usage :
    python -m lra.retrieval.rerank "durée maximale d'un accord-cadre" [--config bge-m3] [-k 10]

Les candidats viennent de la recherche hybride (CANDIDATES premiers). Le modèle par défaut est
BAAI/bge-reranker-v2-m3 (Apache-2.0, multilingue). Son score brut passe par une sigmoïde pour
tomber entre 0 et 1, comme le suggère sa fiche : ce score servira au seuil de refus.
La fiche utilise max_length=512 dans ses exemples : les articles plus longs sont tronqués.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from typing import Protocol

import psycopg

from lra.config import database_url
from lra.embeddings.models import CONFIGS, document_text
from lra.retrieval import hybrid
from lra.retrieval.hybrid import FusedHit

RERANKER_ID = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512
CANDIDATES = 30


class Scorer(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Scores bruts (logits), un par document."""
        ...


@dataclass(frozen=True)
class RerankedHit:
    fused: FusedHit  # résultat hybride, rang mis à jour après reranking
    hybrid_rank: int  # rang avant reranking
    rerank_score: float  # entre 0 et 1


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def rerank(query: str, candidates: list[FusedHit], scorer: Scorer, k: int) -> list[RerankedHit]:
    if not candidates:
        return []
    docs = [document_text(c.hit.num, c.hit.path, c.hit.text) for c in candidates]
    scores = [sigmoid(s) for s in scorer.score(query, docs)]
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].hit.rank))[:k]
    return [
        RerankedHit(
            replace(candidates[i], hit=replace(candidates[i].hit, rank=r)), candidates[i].hit.rank, scores[i]
        )
        for r, i in enumerate(order, start=1)
    ]


class CrossEncoderScorer:
    """Scorer réel. Nécessite torch et sentence-transformers."""

    def __init__(self, model_id: str = RERANKER_ID, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, max_length=MAX_LENGTH, device=device)
        if self.model.model.device.type == "cuda":
            self.model.model.half()

    def score(self, query: str, documents: list[str]) -> list[float]:
        import torch

        pairs = [(query, d) for d in documents]
        identity = torch.nn.Identity()
        # On veut les logits bruts : le nom du paramètre dépend de la version de sentence-transformers.
        try:
            raw = self.model.predict(pairs, batch_size=8, activation_fn=identity)
        except TypeError:
            raw = self.model.predict(pairs, batch_size=8, activation_fct=identity)
        return [float(x) for x in raw]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("query")
    p.add_argument("--config", default="bge-m3", choices=sorted(CONFIGS))
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--as-of", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    from lra.embeddings.models import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(a.config, device=a.device)
    scorer = CrossEncoderScorer(device=a.device)
    with psycopg.connect(database_url()) as conn:
        candidates = hybrid.search(conn, encoder, a.query, CANDIDATES, a.as_of)
    started = time.perf_counter()
    results = rerank(a.query, candidates, scorer, a.k)
    ms = (time.perf_counter() - started) * 1000
    if not results:
        print("Aucun résultat.")
    for r in results:
        h, f = r.fused.hit, r.fused
        origine = ", ".join(f"{src} #{n}" for src, n in sorted(f.ranks.items()))
        print(
            f"{h.rank:>2}. {h.num:<12} rerank={r.rerank_score:.3f}  (hybride #{r.hybrid_rank}; {origine})  {h.path}"
        )
        print(f"    {h.text.replace(chr(10), ' ')[:160]}")
    print(f"\nReranking de {len(candidates)} candidats : {ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
