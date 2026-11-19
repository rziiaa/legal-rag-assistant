from pathlib import Path

import pytest

from lra.ingest.parse import clean_text, is_applicable, parse_article

FIXTURES = Path(__file__).parent / "fixtures"


def test_real_article_l1():
    art = parse_article((FIXTURES / "LEGIARTI000037703232.xml").read_bytes())
    assert art.id == "LEGIARTI000037703232"
    assert art.num == "L1"
    assert art.etat == "VIGUEUR"
    assert art.date_debut == "2019-04-01"
    assert art.date_fin is None  # 2999-01-01 = sans fin
    assert art.type == "AUTONOME"
    assert art.code_id == "LEGITEXT000037701019"
    assert art.code_title == "Code de la commande publique"
    assert art.hierarchy == [
        {"id": "LEGISCTA000037703226", "title": "Partie législative"},
        {"id": "LEGISCTA000037703230", "title": "Titre Préliminaire"},
    ]
    assert art.text == (
        "Les acheteurs et les autorités concédantes choisissent librement, pour répondre à "
        "leurs besoins, d'utiliser leurs propres moyens ou d'avoir recours à un contrat de la "
        "commande publique."
    )
    assert art.nota == ""
    assert len(art.versions) == 1 and art.versions[0].date_fin is None


def test_synthetic_modified_version():
    art = parse_article((FIXTURES / "SYNTHETIC_modifie.xml").read_bytes())
    assert art.etat == "MODIFIE"
    assert art.date_fin == "2026-08-19"
    assert [h["title"] for h in art.hierarchy] == [
        "Partie législative",
        "PREMIÈRE PARTIE : TEST",
        "Livre Ier : TEST",
    ]
    assert art.text.splitlines() == [
        "Phrase d'introduction :",
        "1° Premier point ;",
        "2° Second point.",
        "Cellule A Cellule B",
    ]
    assert art.nota == "Note de test."
    assert [(v.etat, v.date_debut, v.date_fin) for v in art.versions] == [
        ("MODIFIE", "2019-04-01", "2026-08-19"),
        ("VIGUEUR", "2026-08-19", None),
    ]


@pytest.mark.parametrize(
    ("debut", "fin", "as_of", "expected"),
    [
        ("2019-04-01", "2026-08-19", "2026-08-18", True),
        ("2019-04-01", "2026-08-19", "2026-08-19", False),  # le jour du remplacement : version suivante
        ("2026-08-19", None, "2026-08-19", True),
        ("2026-08-19", None, "2026-08-18", False),
        ("2020-01-01", "2020-01-01", "2020-01-01", False),  # début == fin : jamais applicable
    ],
)
def test_is_applicable(debut, fin, as_of, expected):
    assert is_applicable(debut, fin, as_of) is expected


def test_stillborn_state_is_never_applicable():
    assert is_applicable("2022-01-01", None, "2026-11-08", "MODIFIE_MORT_NE") is False
    assert is_applicable("2022-01-01", None, "2026-11-08", "VIGUEUR") is True


def test_external_entities_are_not_resolved(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("NE_DOIT_PAS_APPARAITRE", encoding="utf-8")
    xml = (
        f'<?xml version="1.0"?><!DOCTYPE ARTICLE [<!ENTITY x SYSTEM "{secret.as_uri()}">]>'
        "<ARTICLE><BLOC_TEXTUEL><CONTENU><p>&x;</p></CONTENU></BLOC_TEXTUEL></ARTICLE>"
    ).encode()
    art = parse_article(xml)
    assert "NE_DOIT_PAS_APPARAITRE" not in art.text


def test_wrong_root_is_rejected():
    with pytest.raises(ValueError):
        parse_article(b"<TEXTE_VERSION/>")


def test_clean_text_none():
    assert clean_text(None) == ""
