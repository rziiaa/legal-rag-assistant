"""Construit le corpus (JSONL) à partir des articles extraits de l'archive LEGI.

Usage :
    python -m lra.ingest.build --ccp-dir data/ccp --out data/corpus \
        --archive-name Freemium_legi_global_20261108-140000.tar.gz --sha256 <empreinte>

Écrit dans --out : articles.jsonl, corpus_meta.json (mentions exigées par la licence),
audit.json (comptages à relire avant d'utiliser le corpus).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from lxml import etree

from lra.ingest.parse import ArticleVersion, parse_article

ARCHIVE_URL_BASE = "https://echanges.dila.gouv.fr/OPENDATA/LEGI/"
LICENCE = "Licence Ouverte 2.0 (Etalab)"
_ARCHIVE_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})-(\d{6})")
_NUM_PARTS = re.compile(r"(\d+)")


def archive_date(archive_name: str) -> str | None:
    """Date AAAA-MM-JJ lue dans le nom de l'archive (ex. ..._20261108-140000.tar.gz)."""
    m = _ARCHIVE_DATE.search(archive_name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def build_meta(archive_name: str, sha256: str | None) -> dict[str, Any]:
    date = archive_date(archive_name)
    mention = (
        "Source : DILA, base LEGI (Légifrance), données réutilisées sous Licence Ouverte 2.0. "
        f"Téléchargement : {ARCHIVE_URL_BASE} ; fichier {archive_name}" + (f" du {date}." if date else ".")
    )
    return {
        "provider": "DILA - Direction de l'information légale et administrative",
        "database": "LEGI (codes, lois et règlements consolidés)",
        "licence": LICENCE,
        "archive_name": archive_name,
        "archive_url": ARCHIVE_URL_BASE + archive_name,
        "archive_date": date,
        "archive_sha256": sha256,
        "source_mention": mention,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _natural_key(num: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in _NUM_PARTS.split(num))


def build_corpus(ccp_dir: Path, out_dir: Path, archive_name: str, sha256: str | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = build_meta(archive_name, sha256)
    corpus_date = meta["archive_date"] or "0000-00-00"

    articles: list[ArticleVersion] = []
    errors: list[str] = []
    for _path, art in _safe_iter(ccp_dir / "articles", errors):
        articles.append(art)
    articles.sort(key=lambda a: (_natural_key(a.num), a.date_debut, a.id))

    by_etat = collections.Counter(a.etat for a in articles)
    by_partie = collections.Counter(
        (a.hierarchy[0]["title"] if a.hierarchy else "(sans hiérarchie)") for a in articles
    )
    ids = collections.Counter(a.id for a in articles)
    versions_per_num = collections.Counter(collections.Counter(a.num for a in articles).values())
    audit = {
        "articles_total": len(articles),
        "numeros_distincts": len({a.num for a in articles}),
        "par_etat": dict(by_etat.most_common()),
        "par_partie": dict(by_partie.most_common()),
        "nombre_de_versions_par_numero": {str(k): v for k, v in sorted(versions_per_num.items())},
        "id_en_double": [i for i, c in ids.items() if c > 1][:20],
        "texte_vide": sum(1 for a in articles if not a.text),
        "debut_apres_fin": sum(1 for a in articles if a.date_fin and a.date_debut > a.date_fin),
        "debut_egal_fin": sum(1 for a in articles if a.date_fin and a.date_debut == a.date_fin),
        "debut_apres_date_du_corpus": sum(1 for a in articles if a.date_debut > corpus_date),
        "fin_apres_date_du_corpus_et_non_ouverte": sum(
            1 for a in articles if a.date_fin and a.date_fin > corpus_date
        ),
        "autre_code": sorted(
            {a.code_id for a in articles if a.code_id and a.code_id != "LEGITEXT000037701019"}
        ),
        "erreurs_de_lecture": errors[:20],
        "erreurs_de_lecture_total": len(errors),
    }

    with (out_dir / "articles.jsonl").open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / "corpus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _safe_iter(articles_dir: Path, errors: list[str]):
    for path in sorted(articles_dir.glob("*.xml")):
        try:
            yield path, parse_article(path.read_bytes())
        except (etree.XMLSyntaxError, ValueError, OSError) as exc:  # un fichier illisible ne bloque pas tout
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ccp-dir", default="data/ccp", type=Path)
    p.add_argument("--out", default="data/corpus", type=Path)
    p.add_argument("--archive-name", required=True)
    p.add_argument("--sha256", default=None)
    a = p.parse_args()
    audit = build_corpus(a.ccp_dir, a.out, a.archive_name, a.sha256)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\nFichiers ecrits dans {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
