"""Lecture d'un fichier article de la base LEGI (XML DILA) en une version d'article.

Un fichier LEGIARTI... = UNE version d'un article. Les autres versions du même article sont
d'autres fichiers ; le bloc <VERSIONS> les relie. Le critère d'application à une date est
fondé sur les dates de début et de fin, pas sur l'état (voir `is_applicable`).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from lxml import etree

# Valeur utilisée par la DILA pour « pas de date de fin » (observée dans les fichiers).
OPEN_END = "2999-01-01"

_BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
_CELL_TAGS = {"td", "th"}


@dataclass(frozen=True)
class VersionLink:
    id: str
    etat: str
    date_debut: str
    date_fin: str | None


@dataclass(frozen=True)
class ArticleVersion:
    id: str
    num: str
    etat: str
    date_debut: str
    date_fin: str | None  # None = sans date de fin
    type: str
    code_id: str
    code_title: str
    hierarchy: list[dict[str, str]] = field(default_factory=list)  # [{id, title}, ...] du haut vers le bas
    text: str = ""
    nota: str = ""
    versions: list[VersionLink] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parser() -> etree.XMLParser:
    # Fichiers téléchargés = données non fiables : pas d'entités externes, pas de réseau.
    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)


def _end(value: str | None) -> str | None:
    value = (value or "").strip()
    return None if not value or value == OPEN_END else value


def _collect(el: etree._Element, parts: list[str]) -> None:
    if not isinstance(el.tag, str):  # commentaires, instructions de traitement
        return
    tag = el.tag.lower()
    if tag == "br":
        parts.append("\n")
        return
    block = tag in _BLOCK_TAGS
    if block:
        parts.append("\n")
    if el.text:
        parts.append(el.text)
    for child in el:
        _collect(child, parts)
        if child.tail:
            parts.append(child.tail)
    if tag in _CELL_TAGS:
        parts.append(" ")
    if block:
        parts.append("\n")


def clean_text(el: etree._Element | None) -> str:
    """Texte brut d'un bloc XML : <br/> et blocs deviennent des retours à la ligne."""
    if el is None:
        return ""
    parts: list[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        _collect(child, parts)
        if child.tail:
            parts.append(child.tail)
    raw = "".join(parts).replace(" ", " ").replace(" ", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines())
    return "\n".join(line for line in lines if line)


def _text(root: etree._Element, path: str) -> str:
    found = root.find(path)
    return (found.text or "").strip() if found is not None else ""


def parse_article(data: bytes) -> ArticleVersion:
    root = etree.fromstring(data, _parser())
    if root.tag != "ARTICLE":
        raise ValueError(f"racine inattendue : {root.tag}")

    hierarchy: list[dict[str, str]] = []
    texte = root.find("CONTEXTE/TEXTE")
    code_id = texte.get("cid", "") if texte is not None else ""
    code_title = ""
    node = texte
    if texte is not None:
        titre_txt = texte.find("TITRE_TXT")
        code_title = (titre_txt.text or "").strip() if titre_txt is not None else ""
        while node is not None:
            tm = node.find("TM")
            if tm is None:
                break
            titre = tm.find("TITRE_TM")
            if titre is not None:
                hierarchy.append({"id": titre.get("id", ""), "title": (titre.text or "").strip()})
            node = tm

    versions = [
        VersionLink(
            id=lien.get("id", ""),
            etat=lien.get("etat", ""),
            date_debut=lien.get("debut", ""),
            date_fin=_end(lien.get("fin")),
        )
        for lien in root.findall("VERSIONS/VERSION/LIEN_ART")
    ]

    return ArticleVersion(
        id=_text(root, "META/META_COMMUN/ID"),
        num=_text(root, "META/META_SPEC/META_ARTICLE/NUM"),
        etat=_text(root, "META/META_SPEC/META_ARTICLE/ETAT"),
        date_debut=_text(root, "META/META_SPEC/META_ARTICLE/DATE_DEBUT"),
        date_fin=_end(_text(root, "META/META_SPEC/META_ARTICLE/DATE_FIN")),
        type=_text(root, "META/META_SPEC/META_ARTICLE/TYPE"),
        code_id=code_id,
        code_title=code_title,
        hierarchy=hierarchy,
        text=clean_text(root.find("BLOC_TEXTUEL/CONTENU")),
        nota=clean_text(root.find("NOTA/CONTENU")),
        versions=versions,
    )


def is_applicable(date_debut: str, date_fin: str | None, as_of: str) -> bool:
    """Vrai si la version s'applique à `as_of` (dates ISO AAAA-MM-JJ) : début <= date < fin.

    La date de fin est exclue : le jour où une version est remplacée, c'est la suivante qui
    s'applique. Une version dont début == fin n'est jamais applicable. L'état (VIGUEUR,
    MODIFIE...) n'est volontairement pas utilisé.
    """
    return date_debut <= as_of and (date_fin is None or as_of < date_fin)
