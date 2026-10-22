# Esperanto Text Classification

[![Tests](https://github.com/Godalberto10/Esperanto-Text-Classification/actions/workflows/tests.yml/badge.svg)](https://github.com/Godalberto10/Esperanto-Text-Classification/actions/workflows/tests.yml)

Topic classification of Esperanto texts, built end to end: a reproducible
dataset pipeline from Esperanto Wikipedia, a rule-based NLP library with a
spaCy-like API written from scratch, and a classification notebook.

## Topics

World religions:

| Class | Wikipedia category (eo) |
|---|---|
| `kristanismo` | [Kategorio:Kristanismo](https://eo.wikipedia.org/wiki/Kategorio:Kristanismo) |
| `islamo` | [Kategorio:Islamo](https://eo.wikipedia.org/wiki/Kategorio:Islamo) |
| `judismo` | [Kategorio:Judismo](https://eo.wikipedia.org/wiki/Kategorio:Judismo) |

## Project structure

```
src/esperanto_nlp/   Rule-based NLP library
tests/               Unit tests (pytest)
scripts/             Dataset build script
data/                Dataset (UTF-8 CSV, ';' separator, text;class columns)
notebooks/           Training and classification notebook
```

## Installation

```bash
pip install git+https://github.com/Godalberto10/Esperanto-Text-Classification.git
```

## Library usage

```python
import esperanto_nlp

nlp = esperanto_nlp.load()
doc = nlp("La judoj preĝas en la sinagogo. Ĉu vi?")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.morph, token.is_stop)
# La        la        DET
# judoj     judo      NOUN   Case=Nom|Number=Plur
# preĝas    preĝi     VERB   Mood=Ind|Tense=Pres|VerbForm=Fin
# ...

for sent in doc.sents:
    print(sent.text)
```

The pipeline (`nlp.pipe_names`) has three rule-based components:

| Component | Sets | Rules |
|---|---|---|
| `tagger` | `pos_`, `morph` | Word endings (-o, -a, -e, -i, -as...), a closed lexicon of function words and correlatives, and context (DET/PRON, proper nouns) |
| `lemmatizer` | `lemma_` | Strips inflection: plural, accusative and verb tense (`estis` → `esti`). Derivation is kept (`konataj` → `konata`) |
| `stopwords` | `is_stop` | Stop-word list matched against both the surface form and the lemma |

Part-of-speech tags follow [Universal Dependencies](https://universaldependencies.org/u/pos/).

The stop-word list starts from `vortoj.json` in the
[nlp-esperantilo](https://github.com/jparisu/nlp-esperantilo) repository
(Apache-2.0) and is extended by rule with the correlative table and the
inflected pronoun forms.

## Building the dataset

```bash
pip install -e ".[dataset]"
python scripts/build_dataset.py
```

## Development

Editable install with test dependencies:

```bash
pip install -e ".[dev]"
pytest
```
