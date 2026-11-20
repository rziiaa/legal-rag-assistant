"""Génération de la réponse avec citations obligatoires et refus explicite.

Usage :
    python -m lra.generation.answer "Quelle est la durée maximale d'un accord-cadre ?" \
        [--llm qwen3:4b] [--top 5] [--threshold 0.5]

Deux niveaux de refus :
1. avant le modèle : si le meilleur score de reranking est sous --threshold (désactivé par défaut,
   le seuil sera calibré sur la partie dev du jeu de test) ;
2. par le modèle : consigne du prompt, champ "refus" de la sortie JSON.

Contrôle des citations, fait par le code et non par le modèle : un article cité qui ne fait pas
partie des articles fournis est invalide. Une réponse sans aucune citation valide est signalée.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field

from lra.generation.llm import ANSWER_SCHEMA, LLM, MODELS, OllamaLLM
from lra.generation.prompt import Passage, Prompt, build_user_message, load_prompt

_NUM = re.compile(r"\b([LRDA])\s*\.?\s*(\d+(?:-\d+)*)\b")


def normalize_citation(raw: str) -> str:
    """'Article R. 2124-1' -> 'R2124-1'. Laisse tel quel ce qui ne ressemble pas à un numéro."""
    m = _NUM.search(raw.strip())
    return f"{m.group(1)}{m.group(2)}" if m else raw.strip()


@dataclass
class Answer:
    question: str
    refused: bool
    refusal_reason: str  # "", "score_insuffisant", "aucun_passage", "modele", "sortie_invalide"
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    no_valid_source: bool  # réponse donnée sans aucune citation valide
    passages: list[str]  # numéros des articles fournis au modèle
    top_score: float | None
    model: str
    prompt_version: str
    llm_latency_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    model_refusal_reason: str = ""
    raw_output: str = field(default="", repr=False)


def answer_question(
    question: str,
    passages: list[Passage],
    llm: LLM,
    prompt: Prompt,
    top_score: float | None = None,
    threshold: float | None = None,
) -> Answer:
    base = {
        "question": question,
        "passages": [p.num for p in passages],
        "top_score": top_score,
        "model": llm.name,
        "prompt_version": prompt.version,
    }

    def refusal(reason: str, **extra) -> Answer:
        return Answer(
            refused=True,
            refusal_reason=reason,
            answer="",
            citations=[],
            invalid_citations=[],
            no_valid_source=False,
            **base,
            **extra,
        )

    if not passages:
        return refusal("aucun_passage")
    if threshold is not None and top_score is not None and top_score < threshold:
        return refusal("score_insuffisant")

    result = llm.chat(prompt.text, build_user_message(question, passages), ANSWER_SCHEMA)
    timing = {
        "llm_latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "raw_output": result.content,
    }
    try:
        data = json.loads(result.content)
        refused = bool(data["refus"])
        text = str(data.get("reponse", ""))
        cited = [normalize_citation(str(c)) for c in data.get("citations", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        return refusal("sortie_invalide", **timing)

    if refused:
        return refusal("modele", model_refusal_reason=str(data.get("motif_refus", "")), **timing)

    allowed = {p.num for p in passages}
    cited = list(dict.fromkeys(cited))  # sans doublons, ordre conservé
    valid = [c for c in cited if c in allowed]
    invalid = [c for c in cited if c not in allowed]
    return Answer(
        refused=False,
        refusal_reason="",
        answer=text,
        citations=valid,
        invalid_citations=invalid,
        no_valid_source=not valid,
        **base,
        **timing,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("question")
    p.add_argument("--llm", default="qwen3:4b", help=f"modèle Ollama ({', '.join(MODELS)})")
    p.add_argument("--prompt", default="answer_v1")
    p.add_argument("--top", type=int, default=5, help="nombre d'articles fournis au modèle")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--config", default="bge-m3")
    p.add_argument("--as-of", default=None)
    a = p.parse_args()

    import psycopg

    from lra.config import database_url
    from lra.embeddings.models import SentenceTransformerEncoder
    from lra.retrieval import hybrid
    from lra.retrieval.rerank import CANDIDATES, CrossEncoderScorer, rerank

    encoder = SentenceTransformerEncoder(a.config)
    scorer = CrossEncoderScorer()
    with psycopg.connect(database_url()) as conn:
        candidates = hybrid.search(conn, encoder, a.question, CANDIDATES, a.as_of)
    ranked = rerank(a.question, candidates, scorer, a.top)
    passages = [
        Passage(
            r.fused.hit.num, r.fused.hit.path, r.fused.hit.text, r.fused.hit.date_debut, r.fused.hit.date_fin
        )
        for r in ranked
    ]
    top = ranked[0].rerank_score if ranked else None
    ans = answer_question(a.question, passages, OllamaLLM(a.llm), load_prompt(a.prompt), top, a.threshold)
    out = asdict(ans)
    out.pop("raw_output")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
