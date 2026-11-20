"""Client d'API hébergée testé sans réseau : un faux transport HTTP joue le rôle de l'API."""

import json

import httpx
import pytest

from lra.generation.llm import ANSWER_SCHEMA, OllamaLLM, OpenAICompatibleLLM, make_llm

OK_BODY = {
    "choices": [
        {"message": {"content": '{"refus": false, "reponse": "x", "citations": [], "motif_refus": ""}'}}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def client(handler, sleeps=None):
    return OpenAICompatibleLLM(
        "groq",
        "qwen/qwen3.8-27b",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_sends_schema_and_reads_answer():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=OK_BODY)

    r = client(handler).chat("sys", "user", ANSWER_SCHEMA)
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA
    assert seen[0]["temperature"] == 0
    assert r.model == "groq:qwen/qwen3.8-27b" and r.prompt_tokens == 10 and r.output_tokens == 5


def test_falls_back_to_json_mode_when_schema_is_refused():
    formats = []

    def handler(request):
        fmt = json.loads(request.content)["response_format"]["type"]
        formats.append(fmt)
        return httpx.Response(422 if fmt == "json_schema" else 200, json=OK_BODY)

    llm = client(handler)
    llm.chat("sys", "user", ANSWER_SCHEMA)
    llm.chat("sys", "user", ANSWER_SCHEMA)
    assert formats == ["json_schema", "json_object", "json_object"]  # on ne réessaie pas le schéma


def test_retries_on_rate_limit():
    calls = []
    sleeps = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200, json=OK_BODY)

    client(handler, sleeps).chat("sys", "user", ANSWER_SCHEMA)
    assert len(calls) == 3
    assert sleeps.count(3.0) == 2


def test_other_errors_are_raised():
    with pytest.raises(httpx.HTTPStatusError):
        client(lambda request: httpx.Response(401, json={})).chat("sys", "user", ANSWER_SCHEMA)


def test_sends_to_groq_endpoint():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=OK_BODY)

    client(handler).chat("sys", "user", ANSWER_SCHEMA)
    assert urls == ["https://api.groq.com/openai/v1/chat/completions"]


def test_schema_is_strict_compatible():
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == set(ANSWER_SCHEMA["properties"])


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # pas de .env ici
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleLLM("groq", "qwen/qwen3.8-27b")


def test_make_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm = make_llm("groq:qwen/qwen3.8-27b")
    assert isinstance(llm, OpenAICompatibleLLM) and llm.model == "qwen/qwen3.8-27b"
    assert isinstance(make_llm("qwen3:4b"), OllamaLLM)
