"""Client minimal pour un modèle local servi par Ollama (API /api/chat, sortie JSON imposée par schéma)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

DEFAULT_HOST = "http://127.0.0.1:11434"

# Réglages par modèle. think=False coupe le raisonnement affiché de Qwen3 (plus rapide) ;
# None laisse le comportement par défaut du modèle.
MODELS: dict[str, dict] = {
    "qwen3:4b": {"think": False},
    "qwen3:8b": {"think": False},
    "ministral-3:3b": {"think": None},
    "ministral-3:8b": {"think": None},
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "refus": {"type": "boolean"},
        "reponse": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "motif_refus": {"type": "string"},
    },
    "required": ["refus", "reponse", "citations", "motif_refus"],
}


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None


class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str, schema: dict) -> LLMResult: ...


class OllamaLLM:
    def __init__(
        self, model: str, host: str | None = None, num_ctx: int = 8192, timeout: float = 300
    ) -> None:
        self.name = model
        self.host = host or os.environ.get("OLLAMA_HOST_URL", DEFAULT_HOST)
        self.num_ctx = num_ctx
        self.think = MODELS.get(model, {}).get("think")
        self.client = httpx.Client(timeout=timeout)

    def chat(self, system: str, user: str, schema: dict) -> LLMResult:
        body: dict = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            # Température 0 : sortie aussi stable que possible. num_ctx : la valeur par défaut
            # d'Ollama peut être trop petite pour cinq articles.
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            body["think"] = self.think
        started = time.perf_counter()
        r = self.client.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        return LLMResult(
            content=data["message"]["content"],
            model=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )
