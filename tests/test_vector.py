"""Recherche vectorielle avec un encodeur factice (pas de torch, pas de téléchargement).

Les tests sur base sont ignorés si LRA_TEST_DATABASE_URL n'est pas défini, comme pour les
mots-clés. La base doit avoir l'extension pgvector (c'est le cas de l'image Docker du projet).
"""

import hashlib
import math
import os
import shutil
from pathlib import Path

import pytest

from lra.embeddings.models import DIM, document_text, to_pgvector
from lra.ingest.build import build_corpus

FIXTURES = Path(__file__).parent / "fixtures"
TEST_URL = os.environ.get("LRA_TEST_DATABASE_URL")


class FakeEncoder:
    """Vecteur = sac de mots haché : deux textes qui partagent des mots sont proches."""

    config_name = "fake"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


def test_to_pgvector_checks_dimension():
    assert to_pgvector([0.5] * DIM).startswith("[0.5,0.5")
    with pytest.raises(ValueError):
        to_pgvector([0.1, 0.2])


def test_document_text_includes_context():
    assert document_text("L1", "Partie législative", "Texte.") == "Partie législative\nArticle L1\nTexte."


@pytest.mark.skipif(not TEST_URL, reason="LRA_TEST_DATABASE_URL non défini")
def test_index_and_search(tmp_path):
    psycopg = pytest.importorskip("psycopg")
    from lra.db.load import load_corpus
    from lra.embeddings.index import index_articles
    from lra.retrieval.vector import search

    articles = tmp_path / "ccp" / "articles"
    articles.mkdir(parents=True)
    for name in ("LEGIARTI000037703232.xml", "SYNTHETIC_modifie.xml"):
        shutil.copy(FIXTURES / name, articles / name)
    build_corpus(tmp_path / "ccp", tmp_path / "corpus", "Freemium_legi_global_20261108-140000.tar.gz", None)

    enc = FakeEncoder()
    with psycopg.connect(TEST_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS article_embedding, article_version, corpus_meta CASCADE")
        conn.commit()
        load_corpus(conn, tmp_path / "corpus")

        first = index_articles(conn, enc)
        assert first["encodes"] == 2 and first["total_config"] == 2
        assert index_articles(conn, enc)["encodes"] == 0  # déjà fait
        assert index_articles(conn, enc, force=True)["encodes"] == 2

        hits = search(conn, enc, "acheteurs propres moyens commande publique")
        assert [h.num for h in hits] == ["L1"]  # la version synthétique est terminée au 2026-11-08
        assert 0 < hits[0].score <= 1

        old = search(conn, enc, "premier point", as_of="2025-01-01", k=1)
        assert [h.num for h in old] == ["L9999-1"]

        # Recharger le corpus supprime les vecteurs devenus orphelins.
        load_corpus(conn, tmp_path / "corpus")
        assert conn.execute("SELECT count(*) FROM article_embedding").fetchone()[0] == 0
