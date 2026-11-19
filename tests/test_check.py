from lra.ingest.check import check


def art(num, etat, debut, fin, id_=None):
    return {"num": num, "id": id_ or f"{num}-{debut}", "etat": etat, "date_debut": debut, "date_fin": fin}


def test_check_counts_versions_at_date():
    articles = [
        art("L1", "VIGUEUR", "2019-04-01", None),
        art("L2", "MODIFIE", "2019-04-01", "2026-08-19"),
        art("L2", "VIGUEUR", "2026-08-19", None),
        art("L3", "ABROGE", "2019-04-01", "2024-01-01"),
        art("L4", "MODIFIE_MORT_NE", "2025-01-01", "2024-12-31"),  # dates inversées
        art("L5", "VIGUEUR_DIFF", "2027-01-01", None),  # pas encore en vigueur
    ]
    r = check(articles, "2026-11-08")
    assert r["numeros"] == 5
    assert r["numeros_avec_1_version_applicable"] == 2  # L1, L2
    assert r["exemples_sans_version_applicable"] == ["L3", "L4", "L5"]
    assert r["numeros_avec_plusieurs_versions_applicables"] == 0
    assert r["dates_inversees_par_etat"] == {"MODIFIE_MORT_NE": 1}


def test_check_detects_overlap():
    articles = [
        art("L1", "VIGUEUR", "2019-04-01", None, "a"),
        art("L1", "VIGUEUR", "2020-01-01", None, "b"),
    ]
    r = check(articles, "2026-11-08")
    assert r["numeros_avec_plusieurs_versions_applicables"] == 1
    assert [v["id"] for v in r["detail_plusieurs_versions"]["L1"]] == ["a", "b"]
