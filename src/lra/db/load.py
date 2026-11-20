"""Crée le schéma et charge le corpus (articles.jsonl + corpus_meta.json) dans PostgreSQL.

Usage :
    python -m lra.db.load --corpus data/corpus

Le chargement remplace entièrement le contenu précédent, dans une seule transaction.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from lra.config import database_url
from lra.db.applicability import APPLICABLE_SQL, applicable_params

PATH_SEPARATOR = " > "

INSERT_SQL = """
INSERT INTO article_version
    (id, num, etat, date_debut, date_fin, type, code_id, hierarchy, path, text, nota, versions)
VALUES
    (%(id)s, %(num)s, %(etat)s, %(date_debut)s, %(date_fin)s, %(type)s, %(code_id)s,
     %(hierarchy)s, %(path)s, %(text)s, %(nota)s, %(versions)s)
"""


def apply_schema(conn: psycopg.Connection) -> None:
    sql = resources.files("lra.db").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.execute(sql)


def to_row(article: dict) -> dict:
    return {
        **{k: article[k] for k in ("id", "num", "etat", "date_debut", "date_fin", "type", "code_id")},
        "hierarchy": Jsonb(article["hierarchy"]),
        "path": PATH_SEPARATOR.join(h["title"] for h in article["hierarchy"]),
        "text": article["text"],
        "nota": article.get("nota", ""),
        "versions": Jsonb(article["versions"]),
    }


def load_corpus(conn: psycopg.Connection, corpus_dir: Path) -> dict:
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8"))
    with (corpus_dir / "articles.jsonl").open(encoding="utf-8") as f:
        rows = [to_row(json.loads(line)) for line in f if line.strip()]

    with conn.transaction():
        apply_schema(conn)
        conn.execute("DELETE FROM article_version")
        conn.execute("DELETE FROM corpus_meta")
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
        conn.execute(
            "INSERT INTO corpus_meta (key, value) VALUES ('meta', %s)",
            (Jsonb(meta),),
        )

    as_of = meta["archive_date"]
    total = conn.execute("SELECT count(*) FROM article_version").fetchone()[0]
    applicable = conn.execute(
        f"SELECT count(*) FROM article_version WHERE {APPLICABLE_SQL}",  # filtre constant, valeurs en paramètres
        applicable_params(as_of),
    ).fetchone()[0]
    return {"versions_chargees": total, "date_du_corpus": as_of, "versions_applicables": applicable}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corpus", default="data/corpus", type=Path)
    a = p.parse_args()
    with psycopg.connect(database_url()) as conn:
        result = load_corpus(conn, a.corpus)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
