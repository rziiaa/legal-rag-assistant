"""Filtre SQL « version applicable à une date », miroir de lra.ingest.parse.is_applicable."""

from lra.ingest.parse import NEVER_APPLIED_STATES

# Paramètres attendus : %(as_of)s (date) et %(never)s (liste d'états exclus).
APPLICABLE_SQL = (
    "date_debut <= %(as_of)s AND (date_fin IS NULL OR %(as_of)s < date_fin) AND etat <> ALL(%(never)s)"
)


def applicable_params(as_of: str) -> dict:
    return {"as_of": as_of, "never": sorted(NEVER_APPLIED_STATES)}
