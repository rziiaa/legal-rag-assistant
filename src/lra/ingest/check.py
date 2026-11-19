"""Contrôle de cohérence des versions du corpus à une date donnée.

Usage :
    python -m lra.ingest.check --corpus data/corpus [--as-of AAAA-MM-JJ]

Par défaut, la date est celle de l'archive (corpus_meta.json). Pour chaque numéro d'article,
compte les versions applicables à cette date : on attend exactement 0 (article abrogé ou pas
encore en vigueur) ou 1. Plus d'une version applicable est une anomalie à examiner.
Écrit le résultat dans CORPUS/check_<date>.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from lra.ingest.parse import is_applicable


def load_articles(corpus_dir: Path) -> list[dict[str, Any]]:
    with (corpus_dir / "articles.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(articles: list[dict[str, Any]], as_of: str) -> dict[str, Any]:
    by_num: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for a in articles:
        by_num[a["num"]].append(a)

    applicable_by_num = {
        num: [a for a in versions if is_applicable(a["date_debut"], a["date_fin"], as_of)]
        for num, versions in by_num.items()
    }
    multiple = {num: vs for num, vs in applicable_by_num.items() if len(vs) > 1}
    none = sorted(num for num, vs in applicable_by_num.items() if not vs)

    etat_x_applicable: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for a in articles:
        ok = is_applicable(a["date_debut"], a["date_fin"], as_of)
        etat_x_applicable[a["etat"]]["applicable" if ok else "non_applicable"] += 1

    inverted = collections.Counter(
        a["etat"] for a in articles if a["date_fin"] and a["date_debut"] > a["date_fin"]
    )
    vigueur_not_applicable = [
        {"num": a["num"], "id": a["id"], "debut": a["date_debut"], "fin": a["date_fin"]}
        for a in articles
        if a["etat"] == "VIGUEUR" and not is_applicable(a["date_debut"], a["date_fin"], as_of)
    ]

    return {
        "date": as_of,
        "numeros": len(by_num),
        "versions_applicables": sum(len(vs) for vs in applicable_by_num.values()),
        "numeros_avec_1_version_applicable": sum(1 for vs in applicable_by_num.values() if len(vs) == 1),
        "numeros_sans_version_applicable": len(none),
        "exemples_sans_version_applicable": none[:30],
        "numeros_avec_plusieurs_versions_applicables": len(multiple),
        "detail_plusieurs_versions": {
            num: [
                {"id": a["id"], "etat": a["etat"], "debut": a["date_debut"], "fin": a["date_fin"]} for a in vs
            ]
            for num, vs in list(multiple.items())[:20]
        },
        "etat_x_applicable": {k: dict(v) for k, v in sorted(etat_x_applicable.items())},
        "dates_inversees_par_etat": dict(inverted.most_common()),
        "vigueur_non_applicable": vigueur_not_applicable[:20],
        "vigueur_non_applicable_total": len(vigueur_not_applicable),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corpus", default="data/corpus", type=Path)
    p.add_argument("--as-of", default=None)
    a = p.parse_args()
    as_of = a.as_of
    if as_of is None:
        meta = json.loads((a.corpus / "corpus_meta.json").read_text(encoding="utf-8"))
        as_of = meta["archive_date"]
    result = check(load_articles(a.corpus), as_of)
    out = a.corpus / f"check_{as_of}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nResultat ecrit dans {out}")
    return 1 if result["numeros_avec_plusieurs_versions_applicables"] else 0


if __name__ == "__main__":
    sys.exit(main())
