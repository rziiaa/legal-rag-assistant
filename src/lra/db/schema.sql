-- Schéma de la base. Idempotent : peut être rejoué sans effet de bord.

CREATE EXTENSION IF NOT EXISTS unaccent;

-- Configuration plein texte française qui ignore les accents :
-- « autorites concedantes » trouve « autorités concédantes ».
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'lra_french') THEN
        CREATE TEXT SEARCH CONFIGURATION lra_french (COPY = french);
        ALTER TEXT SEARCH CONFIGURATION lra_french
            ALTER MAPPING FOR hword, hword_part, word WITH unaccent, french_stem;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS corpus_meta (
    key   text PRIMARY KEY,
    value jsonb NOT NULL
);

-- Une ligne = une version d'article. date_fin NULL = sans date de fin.
CREATE TABLE IF NOT EXISTS article_version (
    id         text PRIMARY KEY,
    num        text NOT NULL,
    etat       text NOT NULL,
    date_debut date NOT NULL,
    date_fin   date,
    type       text,
    code_id    text NOT NULL,
    hierarchy  jsonb NOT NULL,
    path       text NOT NULL,
    text       text NOT NULL,
    nota       text NOT NULL DEFAULT '',
    versions   jsonb NOT NULL,
    -- Titres de la hiérarchie (poids B) + texte de l'article (poids D).
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('lra_french', path), 'B')
        || setweight(to_tsvector('lra_french', text), 'D')
    ) STORED
);

CREATE INDEX IF NOT EXISTS article_version_tsv_idx ON article_version USING gin (tsv);
CREATE INDEX IF NOT EXISTS article_version_num_idx ON article_version (num);
CREATE INDEX IF NOT EXISTS article_version_dates_idx ON article_version (date_debut, date_fin);

-- Embeddings : un vecteur par version d'article et par configuration (modèle + format du texte).
-- Pas d'index approximatif (HNSW) : avec environ 2 400 vecteurs, la recherche exacte reste
-- rapide et évite qu'une approximation fausse la comparaison des méthodes.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS article_embedding (
    article_id text NOT NULL REFERENCES article_version (id) ON DELETE CASCADE,
    config     text NOT NULL,
    embedding  vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, config)
);
