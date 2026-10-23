# Esperanto Text Classification

[![Tests](https://github.com/Godalberto10/Esperanto-Text-Classification/actions/workflows/tests.yml/badge.svg)](https://github.com/Godalberto10/Esperanto-Text-Classification/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Godalberto10/Esperanto-Text-Classification/blob/main/notebooks/classification.ipynb)

An end-to-end text classification system for **Esperanto**, a language with
almost no NLP tooling. Given a paragraph or a web page written in Esperanto,
the system predicts whether it is about **Christianity**, **Islam** or
**Judaism**.

Everything is built from scratch: the training data is collected
automatically from Esperanto Wikipedia, the linguistic processing is done by
a rule-based NLP library written for this project (no spaCy, NLTK or Stanza),
and a classical machine-learning model is trained on top of it.

## Why Esperanto?

Esperanto is the most widely spoken constructed language, with a large
online community and more than 300,000 Wikipedia articles, yet mainstream NLP
libraries offer little or no support for it.

Its grammar is also fully regular, which makes it a good fit for a rule-based
approach. Every noun ends in `-o`, every adjective in `-a`, every adverb in
`-e`; plural is always `-j`, accusative is always `-n`, and each verb tense
has a single ending with no irregular verbs:

| Word | Analysis | Lemma |
|---|---|---|
| `hundojn` | noun, plural, accusative | `hundo` (dog) |
| `preĝis` | verb, past tense | `preĝi` (to pray) |
| `legita` | adjective, past passive participle | `legita` (read) |

Instead of a statistical model trained on annotated corpora that do not
exist, a small set of grammar rules produces accurate tokenization,
lemmatization and part-of-speech tagging.

## Why these three classes?

The three Abrahamic religions were chosen on purpose because they are
**semantically close**. They share prophets (Abraham, Moses), places
(Jerusalem), concepts (God, prophecy, scripture, prayer) and much of their
history. A classifier cannot rely on a handful of obvious keywords. It has to
pick up subtler vocabulary such as *sinagogo* versus *moskeo* versus
*preĝejo*, or *Torao* versus *Korano* versus *Evangelio*.

| Class | Topic | Source category |
|---|---|---|
| `kristanismo` | Christianity | [Kategorio:Kristanismo](https://eo.wikipedia.org/wiki/Kategorio:Kristanismo) |
| `islamo` | Islam | [Kategorio:Islamo](https://eo.wikipedia.org/wiki/Kategorio:Islamo) |
| `judismo` | Judaism | [Kategorio:Judismo](https://eo.wikipedia.org/wiki/Kategorio:Judismo) |

## How it works

```
 Esperanto Wikipedia ──► 1. Dataset ──► 2. NLP library ──► 3. Classifier ──► web page
   (MediaWiki API)       paragraphs      lemmas, POS,       TF-IDF +          class
                         + labels        stop-words         linear model      probabilities
```

### 1. Dataset: [`scripts/build_dataset.py`](scripts/build_dataset.py)

- Walks each Wikipedia category and its subcategories through the MediaWiki
  API and downloads a reproducible random sample of articles (fixed seed).
- Splits each article into paragraphs. Each paragraph inherits the label of
  its article.
- Cleans the text:
  - drops reference sections ("Vidu ankaŭ", "Referencoj"...), footnote
    markers, URLs and paragraphs shorter than 15 words;
  - drops articles that belong to more than one class, and duplicated
    paragraphs whose labels disagree.
- Writes [`data/dataset.csv`](data/dataset.csv) (UTF-8, `;` separator,
  `text;class` columns) and [`data/sources.csv`](data/sources.csv), which
  lists the article, revision id and paragraph count behind the dataset rows,
  in the same order. This makes the data traceable and lets the model be
  evaluated on articles it has never seen.

Inherited labels are noisy. A paragraph like *"Ĝi estis fondita en 1950"*
("It was founded in 1950") says nothing about religion, yet it carries the
label of its article. The notebook measures how much of the dataset is
like that (see [Results](#results)).

### 2. NLP library: [`src/esperanto_nlp`](src/esperanto_nlp)

An installable Python package with a spaCy-like API:

```python
import esperanto_nlp

nlp = esperanto_nlp.load()
doc = nlp("La judoj preĝas en la sinagogo. Ĉu vi?")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.morph, token.is_stop)
# La        la        DET                                      True
# judoj     judo      NOUN   Case=Nom|Number=Plur              False
# preĝas    preĝi     VERB   Mood=Ind|Tense=Pres|VerbForm=Fin  False
# ...

for sent in doc.sents:
    print(sent.text)
# La judoj preĝas en la sinagogo.
# Ĉu vi?
```

| Component | Sets | How |
|---|---|---|
| Tokenizer | `Doc`, `Token`, `doc.sents` | Regular expressions aware of Esperanto letters (ĉ, ĝ, ŝ...), hyphenated words (`s-ro`), elision (`l'`), abbreviations (`p.K.`, `k.t.p.`) and numbers (`20-a`) |
| `tagger` | `pos_`, `morph` | Word endings, a closed lexicon of function words and correlatives, and context rules (e.g. `tiu` is a determiner before a noun and a pronoun otherwise) |
| `lemmatizer` | `lemma_` | Strips inflection (plural, accusative, tense: `estis` → `esti`) and keeps derivation (`konataj` → `konata`) |
| `stopwords` | `is_stop` | Stop-word list matched against both the surface form and the lemma |

Tags follow [Universal Dependencies](https://universaldependencies.org/u/pos/).
The library also normalizes the "x-system" spelling (`cxiu` → `ĉiu`), which
is common when the keyboard lacks Esperanto letters.

### 3. Classification: [`notebooks/classification.ipynb`](notebooks/classification.ipynb)

A Jupyter notebook, runnable on Google Colab in under a minute on a CPU, that:

1. Loads the dataset and turns each paragraph into lemmas with the NLP
   library, without stop-words or punctuation.
2. Holds out 20% of the **articles** (not paragraphs) as a test set, so the
   model is always evaluated on text from articles it has never seen.
3. Compares TF-IDF + logistic regression, linear SVM and Naive Bayes, on
   lemmas and on raw words, with cross-validation grouped by article. Each
   article is weighted equally so long articles do not dominate.
4. Evaluates the chosen model with accuracy, macro F1 and a confusion
   matrix, inspects its most informative words and analyses its errors.
5. Scrapes real Esperanto web pages from outside Wikipedia and reports the
   probability of each class.

## Results

On 913 test paragraphs from unseen articles:

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF (lemmas, 1-2 grams) + logistic regression | 0.836 | 0.833 |

- Lemmatizing with the library improves macro F1 by about 0.015 over raw
  words for all three models.
- **Label noise, measured:** in an annotated sample of 100 test
  paragraphs ([`data/noise_sample.csv`](data/noise_sample.csv)), 36% say
  nothing about their labelled religion (biographies, dates, geography).
  The model is right on 98% of the on-topic paragraphs and on 61% of the
  off-topic ones, so most errors come from labels that cannot be inferred
  from the text.
- On 7 real pages from Wikisource and Wikinews, 6 are classified correctly,
  including a news story about Hagia Sofia becoming a mosque and a chapter
  of Deuteronomy.

## Project status

| Stage | Status |
|---|---|
| Dataset pipeline | ✅ Done |
| NLP library (tokenizer, lemmatizer, POS tagger, stop-words) | ✅ Done, 124 tests |
| Classification notebook | ✅ Done |
| Web page classification | ✅ Done |

## Getting started

Install the library straight from GitHub:

```bash
pip install git+https://github.com/Godalberto10/Esperanto-Text-Classification.git
```

Rebuild the dataset (takes several minutes because requests are throttled
to respect the Wikipedia API):

```bash
git clone https://github.com/Godalberto10/Esperanto-Text-Classification.git
cd Esperanto-Text-Classification
pip install -e ".[dataset]"
python scripts/build_dataset.py
```

Run the classification notebook in Colab with the badge at the top of this
page, or locally:

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/classification.ipynb
```

Run the tests:

```bash
pip install -e ".[dev]"
pytest
```

### Reproducibility

- Every random step uses seed 42: the article sample, the train/test split,
  the cross-validation folds and the noise sample.
- [`requirements.txt`](requirements.txt) pins the library versions the
  published results were produced with (`pip install -r requirements.txt`),
  and the notebook prints the versions of each run.
- Wikipedia changes every day, so [`data/sources.csv`](data/sources.csv)
  records the revision id of every article used. The notebook loads that
  published dataset by default instead of downloading a new one.

## Project structure

```
├── data/                 Generated dataset and its provenance
├── notebooks/            Training, evaluation and web classification
├── scripts/              Dataset build script
├── src/esperanto_nlp/    Rule-based NLP library
└── tests/                Unit tests (pytest), run on every push by GitHub Actions
```

## Credits

- The base stop-word list comes from `vortoj.json` in
  [nlp-esperantilo](https://github.com/jparisu/nlp-esperantilo) (Apache-2.0),
  which includes [stopwords-iso](https://github.com/stopwords-iso/stopwords-eo)
  (MIT).
- Text from [Esperanto Wikipedia](https://eo.wikipedia.org), available under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## License

Code released under the [Apache License 2.0](LICENSE). The dataset is
derived from Wikipedia and keeps its CC BY-SA 4.0 license.
