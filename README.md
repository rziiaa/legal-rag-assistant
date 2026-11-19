# LexCommande : assistant RAG sur le Code de la commande publique

LexCommande répond aux questions sur le **Code de la commande publique** en citant les articles qui
justifient chaque réponse, et **refuse de répondre** quand il ne trouve pas de source suffisante.

> **Ce n'est pas un conseil juridique.** Les réponses peuvent être incomplètes ou fausses, et les textes
> évoluent. Vérifiez toujours sur [Légifrance](https://www.legifrance.gouv.fr).

Projet personnel de portfolio (IA générative, RAG, évaluation, industrialisation).

## État d'avancement

Le projet est **en cours de construction**. Seule l'ingestion du corpus fonctionne aujourd'hui.

| Jalon | Contenu | État |
| --- | --- | --- |
| 1 | Cadrage, architecture | Fait |
| 2 | Téléchargement, extraction et découpage du corpus par article, avec versions | En cours |
| 3 | PostgreSQL + pgvector, recherche par mots-clés | À faire |
| 4 | Recherche vectorielle puis hybride | À faire |
| 5 | Génération avec citations et refus | À faire |
| 6 | Jeu de test et évaluation comparative | À faire |
| 7 | API FastAPI, Docker, suivi MLflow | À faire |
| 8 | Résultats, CI, publication | À faire |

Aucun résultat chiffré n'est publié pour l'instant : les chiffres viendront uniquement de nos mesures.

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

## Prérequis

- Python 3.11 ou plus
- Git
- Environ 2 Go d'espace disque pour l'archive des textes

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

## Organisation du dépôt

```
scripts/            inspection de l'archive, extraction d'un code
src/lra/ingest/     lecture des articles XML, construction du corpus
src/lra/...         recherche, génération, évaluation, API (à venir)
tests/              tests ; fixtures/ contient un article réel et un fichier synthétique
data/               données locales, non versionnées
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
- Projet personnel, sans lien avec la DILA ni Légifrance, et sans caractère officiel.

## Licences et source des données

- **Code** : licence [MIT](LICENSE).
- **Textes juridiques** : données de la DILA, base LEGI, réutilisées sous
  [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/). Source :
  <https://echanges.dila.gouv.fr/OPENDATA/LEGI/>, fichier `Freemium_legi_global_20261108-140000.tar.gz`
  du 8 novembre 2026. Ces textes ne sont pas couverts par la licence MIT.
- Le fichier `tests/fixtures/LEGIARTI000037703232.xml` est un article réel issu de cette même source.
