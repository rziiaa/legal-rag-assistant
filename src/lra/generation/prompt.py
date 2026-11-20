"""Prompts versionnés : chaque fichier de prompts/ est identifié par son nom et l'empreinte de son contenu."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
MAX_PASSAGE_CHARS = 6000  # au-delà, l'article est tronqué dans le prompt (le plus long en fait 14 613)


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    sha256: str

    @property
    def version(self) -> str:
        return f"{self.name}@{self.sha256[:12]}"


def load_prompt(name: str = "answer_v1", directory: Path = PROMPTS_DIR) -> Prompt:
    text = (directory / f"{name}.md").read_text(encoding="utf-8")
    return Prompt(name, text, hashlib.sha256(text.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class Passage:
    num: str
    path: str
    text: str
    date_debut: str
    date_fin: str | None


def build_user_message(question: str, passages: list[Passage]) -> str:
    blocks = []
    for p in passages:
        text = p.text if len(p.text) <= MAX_PASSAGE_CHARS else p.text[:MAX_PASSAGE_CHARS] + " [...]"
        validite = f"en vigueur depuis le {p.date_debut}" + (f" jusqu'au {p.date_fin}" if p.date_fin else "")
        blocks.append(f"[{p.num}] ({p.path} ; {validite})\n{text}")
    articles = "\n\n".join(blocks) if blocks else "(aucun article)"
    return f"Articles fournis :\n\n{articles}\n\nQuestion : {question}"
