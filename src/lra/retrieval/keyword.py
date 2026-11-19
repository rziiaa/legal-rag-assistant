"""Recherche par mots-clés (plein texte PostgreSQL) : la méthode de référence.

Usage :
    python -m lra.retrieval.keyword "durée maximale d'un accord-cadre" [-k 10] [--as-of AAAA-MM-JJ]

Choix :
- les mots de la question sont reliés par OU, pas par ET : une question en langage courant
  contient des mots absents de l'article, un ET ne renverrait presque rien ;
- le classement utilise ts_rank_cd, qui n'est pas du BM25 ;
- la liste de mots vides français de PostgreSQL ne contient pas « les » : ce mot reste dans la
  requête. On garde ce comportement tel quel pour la référence, l'évaluation dira s'il gêne ;
- seules les versions applicables à la date demandée (par défaut, la date du corpus) sont cherchées.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import psycopg

from lra.config import database_url
from lra.db.applicability import APPLICABLE_SQL, applicable_params

SEARCH_SQL = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('lra_french', %(query)s)::text, ' & ', ' | ')::tsquery AS tsq
)
SELECT id, num, etat, date_debut::text, date_fin::text, path, text, ts_rank_cd(tsv, q.tsq) AS score
FROM article_version, q
WHERE tsv @@ q.tsq AND {APPLICABLE_SQL}
ORDER BY score DESC, num
LIMIT %(k)s
"""  # filtre constant, valeurs passées en paramètres


@dataclass(frozen=True)
class Hit:
    rank: int
    id: str
    num: str
    etat: str
    date_debut: str
    date_fin: str | None
    path: str
    text: str
    score: float


def corpus_date(conn: psycopg.Connection) -> str:
    row = conn.execute("SELECT value->>'archive_date' FROM corpus_meta WHERE key = 'meta'").fetchone()
    if row is None:
        raise RuntimeError("corpus_meta vide : lancer d'abord python -m lra.db.load")
    return row[0]


def search(conn: psycopg.Connection, query: str, k: int = 10, as_of: str | None = None) -> list[Hit]:
    params = {"query": query, "k": k, **applicable_params(as_of or corpus_date(conn))}
    rows = conn.execute(SEARCH_SQL, params).fetchall()
    return [Hit(i, *row[:7], float(row[7])) for i, row in enumerate(rows, start=1)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--as-of", default=None)
    p.add_argument("--json", action="store_true", help="sortie JSON complète")
    a = p.parse_args()
    with psycopg.connect(database_url()) as conn:
        hits = search(conn, a.query, a.k, a.as_of)
    if a.json:
        print(json.dumps([asdict(h) for h in hits], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("Aucun résultat.")
    for h in hits:
        extrait = h.text.replace("\n", " ")[:160]
        print(f"{h.rank:>2}. {h.num:<12} score={h.score:.4f}  {h.path}")
        print(f"    {extrait}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
