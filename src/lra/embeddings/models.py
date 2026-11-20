"""Configurations d'embedding et chargement des modèles.

Une configuration = un modèle + la façon de présenter requêtes et documents. Son nom est
stocké avec chaque vecteur : on peut garder plusieurs configurations côte à côte et les comparer.
Sources des réglages : fiches officielles des modèles sur Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

DIM = 1024  # dimension commune aux trois modèles, imposée par la colonne vector(1024)


@dataclass(frozen=True)
class EmbeddingConfig:
    name: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    query_prompt_name: str | None = None  # prompt intégré au modèle (sentence-transformers)
    max_seq_length: int | None = None


CONFIGS: dict[str, EmbeddingConfig] = {
    # Fiche : 8192 tokens, pas d'instruction nécessaire pour les requêtes.
    "bge-m3": EmbeddingConfig(name="bge-m3", hf_id="BAAI/bge-m3"),
    # Fiche : une instruction sur chaque requête (prompt "query" fourni avec le modèle), rien sur
    # les documents. On limite à 8192 tokens, largement au-dessus de l'article le plus long.
    "qwen3-0.6b": EmbeddingConfig(
        name="qwen3-0.6b",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        query_prompt_name="query",
        max_seq_length=8192,
    ),
    # Fiche : préfixes obligatoires ; 512 tokens max, les articles plus longs sont tronqués.
    "e5-large": EmbeddingConfig(
        name="e5-large",
        hf_id="intfloat/multilingual-e5-large",
        query_prefix="query: ",
        doc_prefix="passage: ",
        max_seq_length=512,
    ),
}


class Encoder(Protocol):
    config_name: str

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


# Au-delà de cette longueur (en caractères), les documents sont encodés un par un pour limiter
# la mémoire GPU : l'attention croît avec le carré de la longueur.
LONG_DOC_CHARS = 2000
BATCH_SIZE = 32


class SentenceTransformerEncoder:
    """Encodeur réel. Nécessite torch et sentence-transformers (installés à part)."""

    def __init__(self, config_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = CONFIGS[config_name]
        self.config_name = config_name
        self.model = SentenceTransformer(self.config.hf_id, device=device)
        if self.config.max_seq_length:
            self.model.max_seq_length = self.config.max_seq_length
        if self.model.device.type == "cuda":
            self.model.half()  # deux fois moins de mémoire GPU, qualité quasi identique

    def _encode(self, texts: list[str], batch_size: int, **kwargs) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, **kwargs
        )
        return [v.astype("float32").tolist() for v in vectors]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.config.doc_prefix + t for t in texts]
        out: list[list[float] | None] = [None] * len(texts)
        short = [i for i, t in enumerate(texts) if len(t) <= LONG_DOC_CHARS]
        long_ = [i for i, t in enumerate(texts) if len(t) > LONG_DOC_CHARS]
        for idx, size in ((short, BATCH_SIZE), (long_, 1)):
            if idx:
                for i, v in zip(idx, self._encode([prefixed[i] for i in idx], size), strict=True):
                    out[i] = v
        return out  # type: ignore[return-value]

    def encode_query(self, text: str) -> list[float]:
        kwargs = {"prompt_name": self.config.query_prompt_name} if self.config.query_prompt_name else {}
        return self._encode([self.config.query_prefix + text], 1, **kwargs)[0]


def document_text(num: str, path: str, text: str) -> str:
    """Texte encodé pour un article : sa place dans le code, son numéro, puis son contenu."""
    return f"{path}\nArticle {num}\n{text}"


def to_pgvector(vector: list[float]) -> str:
    if len(vector) != DIM:
        raise ValueError(f"dimension {len(vector)} au lieu de {DIM}")
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"
