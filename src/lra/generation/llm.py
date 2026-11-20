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
    # Exigé par le mode strict de Groq ; sans effet gênant pour Ollama.
    "additionalProperties": False,
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


# Fournisseurs à API compatible OpenAI (/chat/completions). Clé lue dans la variable indiquée.
PROVIDERS: dict[str, dict[str, str]] = {
    # Doc Groq : décodage contraint par schéma avec strict=true (modèles listés dans sa doc).
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "key_env": "GROQ_API_KEY"},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "key_env": "MISTRAL_API_KEY"},
}


class OpenAICompatibleLLM:
    """Modèle hébergé, appelé via une API compatible OpenAI.

    Sortie : on demande d'abord un schéma JSON strict ; si l'API refuse ce format (400/422),
    on se rabat sur le mode JSON simple, le prompt décrivant déjà la forme attendue.
    Les erreurs 429 (limite de débit) sont retentées avec une attente croissante.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
        min_interval_s: float = 2.1,
        max_retries: int = 6,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        from lra.config import load_dotenv

        load_dotenv()
        conf = PROVIDERS[provider]
        key = api_key or os.environ.get(conf["key_env"])
        if not key:
            raise RuntimeError(f"{conf['key_env']} absent du fichier .env")
        self.name = f"{provider}:{model}"
        self.url = conf["url"]
        self.model = model
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.sleep = sleep
        self.use_schema = True
        self._last_call = 0.0
        self.client = httpx.Client(
            timeout=timeout, transport=transport, headers={"Authorization": f"Bearer {key}"}
        )

    def _post(self, body: dict) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
            self._last_call = time.monotonic()
            r = self.client.post(self.url, json=body)
            if r.status_code != 429 or attempt == self.max_retries:
                return r
            retry_after = r.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else 2.0 * 2**attempt
            except ValueError:
                delay = 2.0 * 2**attempt
            self.sleep(delay)
        return r

    def chat(self, system: str, user: str, schema: dict) -> LLMResult:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }
        started = time.perf_counter()
        r = None
        if self.use_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reponse_juridique", "schema": schema, "strict": True},
            }
            r = self._post(body)
            if r.status_code in (400, 422):
                self.use_schema = False  # format refusé : on n'essaie plus pour ce client
                r = None
        if r is None:
            body["response_format"] = {"type": "json_object"}
            r = self._post(body)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResult(
            content=data["choices"][0]["message"]["content"],
            model=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def make_llm(name: str) -> LLM:
    """'qwen3:4b' -> Ollama local ; 'groq:<modèle>' ou 'mistral:<modèle>' -> API hébergée."""
    provider, _, model = name.partition(":")
    if provider in PROVIDERS and model:
        return OpenAICompatibleLLM(provider, model)
    return OllamaLLM(name)
