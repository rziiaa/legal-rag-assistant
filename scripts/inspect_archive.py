"""Inspecte une archive LEGI (tar.gz) en flux, sans l'extraire sur disque.

Usage :
    python3 scripts/inspect_archive.py CHEMIN/Freemium_legi_global_....tar.gz

Affiche : nombre de fichiers, répartition par préfixe de chemin, quelques exemples de chemins
et les fichiers XML dont le début contient le texte cherché (par défaut « commande publique »).
Bibliothèque standard uniquement. Lecture seule.
"""

import argparse
import collections
import tarfile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("archive")
    p.add_argument("--needle", default="commande publique", help="texte cherché (insensible à la casse)")
    p.add_argument("--depth", type=int, default=4, help="profondeur des préfixes de chemin affichés")
    p.add_argument("--max-matches", type=int, default=30)
    p.add_argument("--max-size", type=int, default=200_000, help="taille max des XML examinés (octets)")
    a = p.parse_args()

    needle = a.needle.lower().encode("utf-8")
    prefixes: collections.Counter[str] = collections.Counter()
    samples: dict[str, str] = {}
    first_paths: list[str] = []
    matches: list[tuple[str, str]] = []
    total = 0

    with tarfile.open(a.archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            total += 1
            if len(first_paths) < 10:
                first_paths.append(member.name)
            key = "/".join(member.name.split("/")[: a.depth])
            prefixes[key] += 1
            samples.setdefault(key, member.name)
            if len(matches) < a.max_matches and member.name.endswith(".xml") and member.size <= a.max_size:
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                head = handle.read(4096)
                if needle in head.lower():
                    matches.append((member.name, head[:300].decode("utf-8", "replace")))
            if total % 200_000 == 0:
                print(f"... {total} fichiers lus", flush=True)

    print(f"\n=== Total de fichiers : {total}")
    print("\n=== Dix premiers chemins")
    for path in first_paths:
        print(" ", path)
    print(f"\n=== Préfixes (profondeur {a.depth}), 60 plus fournis")
    for key, count in prefixes.most_common(60):
        print(f"{count:>9}  {key}   (ex. {samples[key]})")
    print(f"\n=== XML contenant « {a.needle} » (max {a.max_matches})")
    for name, head in matches:
        print("-", name)
        print("   ", " ".join(head.split())[:300])


if __name__ == "__main__":
    main()
