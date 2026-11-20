"""Recherche hybride : fusion des classements mots-clés et vecteurs (Reciprocal Rank Fusion).

Usage :
    python -m lra.retrieval.hybrid "durée maximale d'un accord-cadre" --config bge-m3 [-k 10]

Chaque méthode renvoie ses N meilleurs articles. Un article reçoit 1 / (RRF_K + rang) pour
chaque liste où il apparaît, et on additionne. Seuls les rangs comptent, pas les scores : pas
besoin de rendre comparables un score plein texte et une similarité cosinus. RRF_K = 60 est la
valeur d'usage de la méthode ; elle n'est pas réglée sur nos questions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace

import psycopg

from lra.config import database_url
from lra.embeddings.models import CONFIGS, Encoder
from lra.retrieval import keyword, vector
from lra.retrieval.keyword import Hit

RRF_K = 60
CANDIDATES = 50  # nombre de résultats demandés à chaque méthode avant fusion


@dataclass(frozen=True)
class FusedHit:
    hit: Hit
    ranks: dict[str, int]  # rang dans chaque liste d'origine, ex. {"keyword": 3, "vector": 1}


def rrf(ranked_lists: dict[str, list[Hit]], k: int, rrf_k: int = RRF_K) -> list[FusedHit]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first_seen: dict[str, Hit] = {}
    for source, hits in ranked_lists.items():
        for h in hits:
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (rrf_k + h.rank)
            ranks.setdefault(h.id, {})[source] = h.rank
            first_seen.setdefault(h.id, h)
    ordered = sorted(scores, key=lambda i: (-scores[i], first_seen[i].num))[:k]
    return [
        FusedHit(replace(first_seen[i], rank=r, score=scores[i]), ranks[i])
        for r, i in enumerate(ordered, start=1)
    ]


def search(
    conn: psycopg.Connection,
    encoder: Encoder,
    query: str,
    k: int = 10,
    as_of: str | None = None,
    candidates: int = CANDIDATES,
) -> list[FusedHit]:
    as_of = as_of or keyword.corpus_date(conn)
    lists = {
        "keyword": keyword.search(conn, query, candidates, as_of),
        "vector": vector.search(conn, encoder, query, candidates, as_of),
    }
    return rrf(lists, k)


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
    with psycopg.connect(database_url()) as conn:
        fused = search(conn, encoder, a.query, a.k, a.as_of)
    if not fused:
        print("Aucun résultat.")
    for f in fused:
        h = f.hit
        origine = ", ".join(f"{src} #{r}" for src, r in sorted(f.ranks.items()))
        print(f"{h.rank:>2}. {h.num:<12} rrf={h.score:.4f}  ({origine})  {h.path}")
        print(f"    {h.text.replace(chr(10), ' ')[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
