"""Tests d'intégration : nécessitent une base PostgreSQL jetable.

Ils sont ignorés si LRA_TEST_DATABASE_URL n'est pas défini. ATTENTION : ils suppriment les tables
de cette base. Ne jamais pointer vers la base de travail.
"""

import os
import shutil
from pathlib import Path

import pytest

from lra.ingest.build import build_corpus

psycopg = pytest.importorskip("psycopg")

TEST_URL = os.environ.get("LRA_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="LRA_TEST_DATABASE_URL non défini")

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE = "Freemium_legi_global_20261108-140000.tar.gz"


@pytest.fixture
def conn(tmp_path):
    from lra.db.load import load_corpus

    articles = tmp_path / "ccp" / "articles"
    articles.mkdir(parents=True)
    for name in ("LEGIARTI000037703232.xml", "SYNTHETIC_modifie.xml"):
        shutil.copy(FIXTURES / name, articles / name)
    build_corpus(tmp_path / "ccp", tmp_path / "corpus", ARCHIVE, None)

    with psycopg.connect(TEST_URL) as c:
        c.execute("DROP TABLE IF EXISTS article_embedding, article_version, corpus_meta CASCADE")
        c.commit()
        result = load_corpus(c, tmp_path / "corpus")
        assert result["versions_chargees"] == 2
        # À la date du corpus, seule L1 est applicable (la version synthétique finit le 2026-08-19).
        assert result["versions_applicables"] == 1
        yield c


def test_finds_article_by_its_words(conn):
    from lra.retrieval.keyword import search

    hits = search(conn, "les acheteurs utilisent leurs propres moyens")
    assert [h.num for h in hits] == ["L1"]
    assert hits[0].path == "Partie législative > Titre Préliminaire"


def test_accents_are_ignored(conn):
    from lra.retrieval.keyword import search

    assert [h.num for h in search(conn, "autorites concedantes")] == ["L1"]


def test_date_filter(conn):
    from lra.retrieval.keyword import search

    assert search(conn, "premier point") == []  # version terminée à la date du corpus
    assert [h.num for h in search(conn, "premier point", as_of="2025-01-01")] == ["L9999-1"]


def test_only_stop_words_returns_nothing(conn):
    from lra.retrieval.keyword import search

    assert search(conn, "de la du et") == []


def test_schema_is_idempotent(conn):
    from lra.db.load import apply_schema

    apply_schema(conn)
    apply_schema(conn)
