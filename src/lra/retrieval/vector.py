"""Recherche vectorielle : les articles dont le sens est le plus proche de la question.

Usage :
    python -m lra.retrieval.vector "durée maximale d'un accord-cadre" --config bge-m3 [-k 10]

Score = similarité cosinus (1 = même direction). Mêmes règles de date que la recherche par
mots-clés. Recherche exacte sur tous les vecteurs de la configuration, sans index approximatif.
"""

from __future__ import annotations

import argparse
import sys

import psycopg

from lra.config import database_url
from lra.db.applicability import APPLICABLE_SQL, applicable_params
from lra.embeddings.models import CONFIGS, Encoder, to_pgvector
from lra.retrieval.keyword import Hit, corpus_date, print_hits

SEARCH_SQL = f"""
SELECT a.id, a.num, a.etat, a.date_debut::text, a.date_fin::text, a.path, a.text,
       1 - (e.embedding <=> %(qv)s::vector) AS score
FROM article_embedding e
JOIN article_version a ON a.id = e.article_id
WHERE e.config = %(config)s AND {APPLICABLE_SQL}
ORDER BY e.embedding <=> %(qv)s::vector, a.num
LIMIT %(k)s
"""  # filtre constant, valeurs passées en paramètres


def search(
    conn: psycopg.Connection, encoder: Encoder, query: str, k: int = 10, as_of: str | None = None
) -> list[Hit]:
    params = {
        "qv": to_pgvector(encoder.encode_query(query)),
        "config": encoder.config_name,
        "k": k,
        **applicable_params(as_of or corpus_date(conn)),
    }
    rows = conn.execute(SEARCH_SQL, params).fetchall()
    if (
        not rows
        and not conn.execute(
            "SELECT 1 FROM article_embedding WHERE config = %s LIMIT 1", (encoder.config_name,)
        ).fetchone()
    ):
        raise RuntimeError(f"aucun vecteur pour {encoder.config_name} : lancer lra.embeddings.index")
    return [Hit(i, *row[:7], float(row[7])) for i, row in enumerate(rows, start=1)]


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
        print_hits(search(conn, encoder, a.query, a.k, a.as_of))
    return 0


if __name__ == "__main__":
    sys.exit(main())
