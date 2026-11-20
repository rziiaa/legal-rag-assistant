# LexCommande : assistant RAG sur le Code de la commande publique

LexCommande répond aux questions sur le **Code de la commande publique** en citant les articles qui
justifient chaque réponse, et **refuse de répondre** quand il ne trouve pas de source suffisante.

> **Ce n'est pas un conseil juridique.** Les réponses peuvent être incomplètes ou fausses, et les textes
> évoluent. Vérifiez toujours sur [Légifrance](https://www.legifrance.gouv.fr).

Projet personnel de portfolio (IA générative, RAG, évaluation, industrialisation).

## État d'avancement

Le projet est **en cours de construction**. Le pipeline complet fonctionne en ligne de commande, de la
recherche à la réponse citée. L'évaluation chiffrée, l'API web et la CI restent à faire.

| Jalon | Contenu | État |
| --- | --- | --- |
| 1 | Cadrage, architecture | Fait |
| 2 | Téléchargement, extraction et découpage du corpus par article, avec versions | Fait |
| 3 | PostgreSQL, recherche par mots-clés | Fait |
| 4 | Recherche vectorielle (3 modèles), hybride, reranking | Fait |
| 5 | Génération avec citations et refus, modèle local ou API | Fait, seuil de refus à calibrer |
| 6 | Jeu de test et évaluation comparative | En cours |
| 7 | API FastAPI, Docker, suivi MLflow | À faire |
| 8 | Résultats, CI, publication | À faire |

Aucun résultat chiffré n'est publié pour l'instant : les chiffres viendront uniquement de nos mesures,
sur un jeu de test écrit pour le projet.

## Comment ça marche (vue d'ensemble)

1. **Préparer** : on télécharge les textes officiels (base LEGI de la DILA), on extrait le Code de la
   commande publique, on le découpe **article par article** en gardant **toutes les versions** et leurs dates.
2. **Chercher** : la question est comparée aux articles par mots-clés et par sens, puis les meilleurs
   résultats sont reclassés.
3. **Décider** : si les articles trouvés ne suffisent pas, l'assistant refuse.
4. **Rédiger** : sinon, un modèle de langage répond en s'appuyant uniquement sur ces articles.
5. **Contrôler** : un programme vérifie que chaque article cité fait partie des articles retrouvés.

Par défaut, seule la version d'un article applicable à la date du corpus est utilisée. Une version
s'applique à une date `d` si `date_debut <= d < date_fin`, sauf les versions « mort-nées »
(`MODIFIE_MORT_NE`), jamais entrées en vigueur, qui sont toujours exclues. Le contrôle
`python -m lra.ingest.check` vérifie qu'aucun article n'a plus d'une version applicable.

## Architecture technique

| Étape | Choix | Remarque |
| --- | --- | --- |
| Corpus | Archive LEGI de la DILA, 1 article = 1 passage | Toutes les versions gardées, filtre par date |
| Stockage | PostgreSQL 17 + pgvector, dans Docker | Une seule base pour texte, plein texte et vecteurs |
| Mots-clés | Plein texte PostgreSQL (français, accents ignorés), mots reliés par OU | Score `ts_rank_cd`, ce n'est pas du BM25 |
| Vecteurs | `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B`, `intfloat/multilingual-e5-large` | Recherche exacte (cosinus), sans index approximatif |
| Hybride | Reciprocal Rank Fusion (k = 60) | Fusion par rangs, pas par scores |
| Reranking | `BAAI/bge-reranker-v2-m3` sur les 30 meilleurs candidats | Score ramené entre 0 et 1 |
| Génération | Local via Ollama (`qwen3:4b`, `ministral-3:3b`) ou API Groq | Sortie JSON imposée par schéma |
| Contrôle | Code Python, pas le modèle | Un article cité absent des passages fournis est invalide |

## Prérequis

- Python 3.11 ou plus, Git, Docker
- Environ 2 Go d'espace disque pour l'archive des textes, et environ 6 Go pour les modèles
- Pour les embeddings et le reranking : une carte graphique NVIDIA est conseillée (testé sur RTX 3060 Ti,
  8 Go), le processeur seul fonctionne mais plus lentement
- Pour la génération locale : [Ollama](https://ollama.com) ; pour l'option API : une clé
  [Groq](https://console.groq.com)

## Installation

```bash
git clone https://github.com/rziiaa/legal-rag-assistant.git
cd legal-rag-assistant
python -m venv .venv
```

Linux / macOS :

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Windows (PowerShell) :

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Dans la suite, remplacez `python` par le Python de votre environnement virtuel.

## Construire le corpus

Les données ne sont pas versionnées dans le dépôt : chacun les télécharge à la source.

**1. Télécharger l'archive globale LEGI** (environ 1,2 Go). Le nom du fichier change à chaque nouveau
stock : consultez la liste sur <https://echanges.dila.gouv.fr/OPENDATA/LEGI/>.

```bash
mkdir -p data/raw
curl -L -C - -o data/raw/Freemium_legi_global_20261108-140000.tar.gz \
  https://echanges.dila.gouv.fr/OPENDATA/LEGI/Freemium_legi_global_20261108-140000.tar.gz
```

Sous PowerShell, utilisez `curl.exe` au lieu de `curl`.

**2. Extraire le Code de la commande publique** (lit l'archive deux fois, plusieurs minutes) :

```bash
python scripts/extract_code.py data/raw/Freemium_legi_global_20261108-140000.tar.gz --out data/ccp
```

Le script repère le code par son titre, copie ses fichiers dans `data/ccp/` et écrit un rapport dans
`data/ccp/report.txt`.

**3. Construire le corpus** :

```bash
python -m lra.ingest.build --ccp-dir data/ccp --out data/corpus \
  --archive-name Freemium_legi_global_20261108-140000.tar.gz --sha256 <empreinte SHA-256 de l'archive>
```

Fichiers produits dans `data/corpus/` :

| Fichier | Contenu |
| --- | --- |
| `articles.jsonl` | Une ligne par version d'article : numéro, état, dates, hiérarchie, texte, note, liens vers les autres versions |
| `corpus_meta.json` | Source, licence, nom, date et empreinte de l'archive |
| `audit.json` | Comptages de contrôle à relire avant d'utiliser le corpus |

Contrôle des versions à la date du corpus (aucun article ne doit avoir deux versions applicables) :

```bash
python -m lra.ingest.check --corpus data/corpus
```

## Base de données et recherche

**1. Démarrer PostgreSQL et charger le corpus.** Copier `.env.example` en `.env` ; si le port 5432 est
déjà pris, changer `POSTGRES_PORT` et le port de `DATABASE_URL`. Sous Windows, garder `127.0.0.1` dans
`DATABASE_URL` : avec `localhost`, chaque connexion peut prendre plusieurs dizaines de secondes.

```bash
docker compose up -d
python -m lra.db.load --corpus data/corpus
```

**2. Installer les modèles d'embedding.** PyTorch s'installe à part, selon la machine : voir
<https://pytorch.org/get-started/locally/>. Puis :

```bash
python -m pip install -e ".[dev,embed]"
python -m lra.embeddings.index --config bge-m3
```

**3. Chercher :**

```bash
python -m lra.retrieval.keyword "durée maximale d'un accord-cadre"
python -m lra.retrieval.vector "durée maximale d'un accord-cadre" --config bge-m3
python -m lra.retrieval.hybrid "durée maximale d'un accord-cadre"
python -m lra.retrieval.rerank "durée maximale d'un accord-cadre"
```

## Générer une réponse citée

Modèle local, après `ollama pull qwen3:4b` :

```bash
python -m lra.generation.answer "Quelle est la durée maximale d'un accord-cadre ?" --llm qwen3:4b
```

Option API, après avoir mis `GROQ_API_KEY` dans `.env` :

```bash
python -m lra.generation.answer "Quelle est la durée maximale d'un accord-cadre ?" --llm groq:qwen/qwen3.8-27b
```

La sortie JSON indique la réponse, les articles cités, les citations invalides, un éventuel refus et son
motif, la version du prompt, le modèle, la latence et le nombre de tokens.

## Tests

```bash
python -m pytest -q
```

Les tests d'intégration demandent une base jetable, jamais la base de travail :

```bash
docker compose exec db createdb -U lra lra_test
LRA_TEST_DATABASE_URL=postgresql://lra:lra@127.0.0.1:5432/lra_test python -m pytest -q
```

Sous PowerShell : `$env:LRA_TEST_DATABASE_URL = "..."` puis `pytest`.

## Organisation du dépôt

```
scripts/              inspection de l'archive, extraction d'un code
prompts/              prompts versionnés
src/lra/ingest/       lecture des articles XML, construction et contrôle du corpus
src/lra/db/           schéma PostgreSQL, chargement
src/lra/embeddings/   configurations des modèles, calcul des vecteurs
src/lra/retrieval/    mots-clés, vecteurs, hybride, reranking
src/lra/generation/   prompt, clients LLM (Ollama, API), réponse citée et refus
tests/                tests ; fixtures/ contient un article réel et un fichier synthétique
data/                 données locales, non versionnées
```

## Contribuer

- Une branche par sujet, créée depuis `dev` : `feat/...`, `fix/...`, `docs/...`, `test/...`, `chore/...`.
- Messages de commit au format `type: description` (ex. `feat: add keyword search`).
- Pull request vers `dev` ; `main` correspond aux versions publiées.
- Les tests doivent passer (`pytest`), ainsi que le linter (`ruff check src tests scripts`).

## Limites

- Périmètre : Code de la commande publique (parties législative et réglementaire, annexes), sans les
  textes liés (CCAG, arrêtés).
- Le corpus est figé à la date de l'archive téléchargée ; les modifications ultérieures n'y sont pas.
- Le contrôle des citations vérifie qu'un article cité fait partie des articles fournis, pas qu'il
  appuie réellement la phrase qui le cite.
- Les articles de plus de 512 tokens sont tronqués pour `multilingual-e5-large` et pour le reranker.
- L'option API envoie la question et les articles retrouvés à un service tiers (Groq).
- Projet personnel, sans lien avec la DILA ni Légifrance, et sans caractère officiel.

## Licences et source des données

- **Code** : licence [MIT](LICENSE).
- **Textes juridiques** : données de la DILA, base LEGI, réutilisées sous
  [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/). Source :
  <https://echanges.dila.gouv.fr/OPENDATA/LEGI/>, fichier `Freemium_legi_global_20261108-140000.tar.gz`
  du 8 novembre 2026. Ces textes ne sont pas couverts par la licence MIT.
- Le fichier `tests/fixtures/LEGIARTI000037703232.xml` est un article réel issu de cette même source.
