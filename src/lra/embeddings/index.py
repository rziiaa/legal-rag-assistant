"""Calcule les embeddings de toutes les versions d'articles et les enregistre dans PostgreSQL.

Usage :
    python -m lra.embeddings.index --config bge-m3 [--device cuda] [--force]

Toutes les versions sont encodées, pas seulement celles en vigueur : le filtre de date se fait
au moment de la recherche. Sans --force, les versions déjà encodées pour cette configuration
sont ignorées.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import psycopg

from lra.config import database_url
from lra.db.load import apply_schema
from lra.embeddings.models import CONFIGS, Encoder, document_text, to_pgvector

UPSERT_SQL = """
INSERT INTO article_embedding (article_id, config, embedding)
VALUES (%s, %s, %s::vector)
ON CONFLICT (article_id, config) DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now()
"""


def index_articles(conn: psycopg.Connection, encoder: Encoder, force: bool = False) -> dict:
    apply_schema(conn)
    cfg = encoder.config_name
    if force:
        conn.execute("DELETE FROM article_embedding WHERE config = %s", (cfg,))
    rows = conn.execute(
        """
        SELECT a.id, a.num, a.path, a.text FROM article_version a
        WHERE NOT EXISTS (
            SELECT 1 FROM article_embedding e WHERE e.article_id = a.id AND e.config = %s
        )
        ORDER BY a.id
        """,
        (cfg,),
    ).fetchall()
    started = time.perf_counter()
    vectors = encoder.encode_documents([document_text(num, path, text) for _, num, path, text in rows])
    seconds = time.perf_counter() - started
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, [(r[0], cfg, to_pgvector(v)) for r, v in zip(rows, vectors, strict=True)])
    conn.commit()
    total = conn.execute("SELECT count(*) FROM article_embedding WHERE config = %s", (cfg,)).fetchone()[0]
    return {
        "config": cfg,
        "encodes": len(rows),
        "secondes_encodage": round(seconds, 1),
        "total_config": total,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--device", default=None, help="cuda, cpu (défaut : automatique)")
    p.add_argument("--force", action="store_true", help="recalculer tous les vecteurs de cette config")
    a = p.parse_args()

    from lra.embeddings.models import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(a.config, device=a.device)
    print(f"Modele charge sur {encoder.model.device}", flush=True)
    with psycopg.connect(database_url()) as conn:
        result = index_articles(conn, encoder, force=a.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
