import json

import pytest

from lra.generation.answer import answer_question, normalize_citation
from lra.generation.llm import LLMResult
from lra.generation.prompt import Passage, build_user_message, load_prompt

PASSAGES = [
    Passage("R2124-1", "Partie réglementaire > Procédure formalisée", "Texte A.", "2019-04-01", None),
    Passage("L2124-1", "Partie législative > Procédure formalisée", "Texte B.", "2019-04-01", "2030-01-01"),
]


class FakeLLM:
    name = "fake"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, system, user, schema):
        self.calls += 1
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResult(content, self.name, 12.0, 100, 20)


@pytest.fixture
def prompt():
    return load_prompt("answer_v1")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R2124-1", "R2124-1"),
        ("Article R. 2124-1", "R2124-1"),
        ("art. L 2125-1", "L2125-1"),
        ("D2196-7", "D2196-7"),
        ("Annexe préliminaire", "Annexe préliminaire"),
    ],
)
def test_normalize_citation(raw, expected):
    assert normalize_citation(raw) == expected


def test_prompt_version_is_stable(prompt):
    assert prompt.version.startswith("answer_v1@")
    assert prompt.version == load_prompt("answer_v1").version


def test_user_message_lists_articles_with_validity():
    msg = build_user_message("Q ?", PASSAGES)
    assert "[R2124-1] (Partie réglementaire > Procédure formalisée ; en vigueur depuis le 2019-04-01)" in msg
    assert "jusqu'au 2030-01-01" in msg
    assert msg.endswith("Question : Q ?")


def test_valid_answer(prompt):
    llm = FakeLLM({"refus": False, "reponse": "Oui.", "citations": ["Article R. 2124-1"], "motif_refus": ""})
    a = answer_question("Q ?", PASSAGES, llm, prompt, top_score=0.9)
    assert not a.refused and a.citations == ["R2124-1"]
    assert a.invalid_citations == [] and not a.no_valid_source
    assert a.passages == ["R2124-1", "L2124-1"] and a.llm_latency_ms == 12.0


def test_invented_citation_is_flagged(prompt):
    llm = FakeLLM({"refus": False, "reponse": "Oui.", "citations": ["R9999-9"], "motif_refus": ""})
    a = answer_question("Q ?", PASSAGES, llm, prompt)
    assert a.citations == [] and a.invalid_citations == ["R9999-9"] and a.no_valid_source


def test_model_refusal(prompt):
    llm = FakeLLM({"refus": True, "reponse": "", "citations": [], "motif_refus": "Rien sur ce sujet."})
    a = answer_question("Q ?", PASSAGES, llm, prompt)
    assert a.refused and a.refusal_reason == "modele" and a.model_refusal_reason == "Rien sur ce sujet."


def test_threshold_refuses_without_calling_model(prompt):
    llm = FakeLLM({"refus": False, "reponse": "x", "citations": ["R2124-1"], "motif_refus": ""})
    a = answer_question("Q ?", PASSAGES, llm, prompt, top_score=0.2, threshold=0.5)
    assert a.refused and a.refusal_reason == "score_insuffisant" and llm.calls == 0


def test_no_passage_refuses(prompt):
    a = answer_question("Q ?", [], FakeLLM({}), prompt)
    assert a.refused and a.refusal_reason == "aucun_passage"


def test_invalid_json_is_a_refusal(prompt):
    a = answer_question("Q ?", PASSAGES, FakeLLM("pas du json"), prompt)
    assert a.refused and a.refusal_reason == "sortie_invalide"
