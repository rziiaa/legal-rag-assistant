"""Extrait un code (par défaut : Code de la commande publique) de l'archive LEGI globale.

Usage :
    python scripts/extract_code.py ARCHIVE.tar.gz [--out data/ccp] [--needle "code de la commande publique"]

Deux passes en flux sur l'archive (qui n'est jamais modifiée) :
  1. lit les fichiers texte/version et texte/struct des codes en vigueur, repère le code dont le
     titre contient le texte cherché, lit sa structure et en tire les identifiants d'articles ;
  2. copie les fichiers de ce code (structure, version, articles, sections, autres) dans
     un dossier plat, avec un index des chemins d'origine.

Écrit aussi OUT/report.txt (UTF-8) : à m'envoyer tel quel. Bibliothèque standard uniquement.
Les noms de fichiers écrits sont toujours le nom de base assaini : aucun chemin de l'archive
n'est utilisé pour écrire sur le disque.
"""

import argparse
import collections
import os
import re
import sys
import tarfile

MARKER = "code_et_TNC_en_vigueur/code_en_vigueur/"
ARTI_ID = re.compile(rb"LEGIARTI\d{12}")
ETAT = re.compile(rb"<ETAT>([^<]*)</ETAT>")
SAFE = re.compile(r"[^A-Za-z0-9._-]")


def pass1(archive: str, max_bytes: int):
    versions: dict[str, tuple[str, bytes]] = {}
    structs: dict[str, tuple[str, bytes]] = {}
    total = 0
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            total += 1
            if total % 500_000 == 0:
                print(f"  passe 1 : {total} fichiers lus", flush=True)
            n = m.name
            if MARKER not in n or not n.endswith(".xml") or m.size > max_bytes:
                continue
            is_version = "/texte/version/" in n
            is_struct = "/texte/struct/" in n
            if not (is_version or is_struct):
                continue
            handle = tar.extractfile(m)
            if handle is None:
                continue
            key = os.path.basename(n)[:-4]
            (versions if is_version else structs)[key] = (n, handle.read())
    return versions, structs, total


def kind_of(name: str, base: str) -> str:
    if "/texte/struct/" in name:
        return "struct"
    if "/texte/version/" in name:
        return "version"
    if base.startswith("LEGIARTI"):
        return "articles"
    if base.startswith("LEGISCTA"):
        return "sections"
    return "other"


def head(data: bytes, n: int) -> str:
    return data[:n].decode("utf-8", "replace")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("archive")
    p.add_argument("--out", default="data/ccp")
    p.add_argument("--needle", default="code de la commande publique")
    p.add_argument("--legitext", default=None, help="forcer l'identifiant LEGITEXT choisi")
    p.add_argument("--max-meta-bytes", type=int, default=60_000_000)
    a = p.parse_args()

    needle = a.needle.lower().encode("utf-8")
    lines: list[str] = []

    def log(s: str = "") -> None:
        lines.append(s)
        print(s.encode("ascii", "replace").decode("ascii"), flush=True)

    log("== Passe 1 : recherche du code")
    versions, structs, total1 = pass1(a.archive, a.max_meta_bytes)
    log(
        f"fichiers lus : {total1} ; versions de codes en vigueur : {len(versions)} ; structures : {len(structs)}"
    )

    candidates = []
    for key, (path, data) in versions.items():
        pos = data[:30_000].lower().find(needle)
        if pos >= 0:
            snippet = " ".join(head(data[max(0, pos - 80) : pos + 160], 240).split())
            candidates.append((key, path, snippet))
    log(f"candidats dont le debut contient '{a.needle}' : {len(candidates)}")
    for key, path, snippet in candidates:
        log(f"- {key}  {path}\n    {snippet}")

    legitext = a.legitext or (candidates[0][0] if candidates else None)
    if not legitext:
        log("Aucun candidat : relancer avec --needle ou --legitext.")
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return 1
    log(f"code retenu : {legitext}")

    struct_data = structs.get(legitext, (None, b""))[1]
    wanted_ids = {m.decode() for m in ARTI_ID.findall(struct_data)}
    log(f"structure : {len(struct_data)} octets ; identifiants d'articles cites : {len(wanted_ids)}")

    log("== Passe 2 : extraction")
    for sub in ("struct", "version", "articles", "sections", "other"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    folder = f"/{legitext}/"
    counts: collections.Counter[str] = collections.Counter()
    etats: collections.Counter[str] = collections.Counter()
    found_ids: set[str] = set()
    samples: dict[str, bytes] = {}
    index_rows: list[str] = []
    other_n = 0
    total2 = 0
    with tarfile.open(a.archive, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            total2 += 1
            if total2 % 500_000 == 0:
                print(f"  passe 2 : {total2} fichiers lus", flush=True)
            n = m.name
            base = os.path.basename(n)
            stem = base.removesuffix(".xml")
            if folder not in n and stem not in wanted_ids:
                continue
            handle = tar.extractfile(m)
            if handle is None:
                continue
            data = handle.read()
            kind = kind_of(n, base)
            safe = SAFE.sub("_", base)
            if kind == "other":
                other_n += 1
                safe = f"{other_n}_{safe}"
            with open(os.path.join(a.out, kind, safe), "wb") as f:
                f.write(data)
            counts[kind] += 1
            index_rows.append(f"{kind}\t{safe}\t{n}")
            if kind == "articles":
                found_ids.add(stem)
                mt = ETAT.search(data)
                etat = mt.group(1).decode() if mt else "?"
                etats[etat] += 1
                if etat not in samples and len(samples) < 4:
                    samples[etat] = data
            elif kind in ("struct", "version") and kind not in samples:
                samples[kind] = data
    with open(os.path.join(a.out, "index.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_rows) + "\n")

    log(f"fichiers lus : {total2}")
    log("fichiers copies par type : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    missing = wanted_ids - found_ids
    log(
        f"articles cites par la structure : {len(wanted_ids)} ; retrouves : {len(wanted_ids & found_ids)} ; absents : {len(missing)}"
    )
    if missing:
        log("exemples d'absents : " + ", ".join(sorted(missing)[:10]))
    log("etats des articles copies : " + ", ".join(f"{k}={v}" for k, v in etats.most_common()))
    log("articles copies hors des identifiants de la structure : " + str(len(found_ids - wanted_ids)))

    log("\n== Extraits (debut des fichiers)")
    for label in ("version", "struct"):
        if label in samples:
            log(f"\n--- {label} ({len(samples[label])} octets)\n{head(samples[label], 2500)}")
    for etat, data in samples.items():
        if etat in ("version", "struct"):
            continue
        log(f"\n--- article etat={etat} ({len(data)} octets)\n{head(data, 3000)}")

    with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRapport ecrit dans {os.path.join(a.out, 'report.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
