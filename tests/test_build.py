import json
import shutil
from pathlib import Path

from lra.ingest.build import archive_date, build_corpus, build_meta

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE = "Freemium_legi_global_20261108-140000.tar.gz"


def test_archive_date():
    assert archive_date(ARCHIVE) == "2026-11-08"
    assert archive_date("sans_date.tar.gz") is None


def test_meta_contains_licence_mentions():
    meta = build_meta(ARCHIVE, "abc")
    # Mentions exigées : paternité (DILA), URL, nom du fichier, date du fichier.
    assert "DILA" in meta["source_mention"]
    assert "https://echanges.dila.gouv.fr/OPENDATA/LEGI/" in meta["source_mention"]
    assert ARCHIVE in meta["source_mention"]
    assert "2026-11-08" in meta["source_mention"]
    assert meta["archive_sha256"] == "abc"


def test_build_corpus(tmp_path):
    articles = tmp_path / "ccp" / "articles"
    articles.mkdir(parents=True)
    for name in ("LEGIARTI000037703232.xml", "SYNTHETIC_modifie.xml"):
        shutil.copy(FIXTURES / name, articles / name)
    (articles / "broken.xml").write_bytes(b"<ARTICLE><oops></ARTICLE>")

    out = tmp_path / "corpus"
    audit = build_corpus(tmp_path / "ccp", out, ARCHIVE, None)

    assert audit["articles_total"] == 2
    assert audit["par_etat"] == {"VIGUEUR": 1, "MODIFIE": 1}
    assert audit["erreurs_de_lecture_total"] == 1
    lines = (out / "articles.jsonl").read_text(encoding="utf-8").splitlines()
    nums = [json.loads(line)["num"] for line in lines]
    assert nums == ["L1", "L9999-1"]  # tri naturel par numéro
    meta = json.loads((out / "corpus_meta.json").read_text(encoding="utf-8"))
    assert meta["archive_date"] == "2026-11-08"
