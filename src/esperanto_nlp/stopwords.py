"""Esperanto stop-word handling.

The base list comes from `data/vortoj.json` (nlp-esperantilo repository,
Apache-2.0, which in turn includes stopwords-iso, MIT). It is extended by rule
with the correlative table and the inflected pronoun forms.
"""

import json
from importlib import resources

PRONOUNS = ["mi", "ci", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si"]

# Correlative table: 5 prefixes x 9 endings (kiu, tiu, iu, ĉiu, neniu...).
CORRELATIVE_PREFIXES = ["ki", "ti", "i", "ĉi", "neni"]
CORRELATIVE_SUFFIXES = ["u", "o", "a", "e", "am", "al", "el", "om", "es"]


def _load_vortoj():
    path = resources.files("esperanto_nlp") / "data" / "vortoj.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)["vortoj"]


# Function word -> resource category (prepozicio, konjunkcio, pronomo...).
FUNCTION_WORDS = {entry["vorto"]: entry["kategorio"] for entry in _load_vortoj()}


def _inflections(word):
    """Plural (-j) and accusative (-n) forms a function word can take."""
    if word.endswith(("u", "a")):
        return [word, word + "j", word + "n", word + "jn"]
    if word.endswith(("o", "e")):
        return [word, word + "n"]
    return [word]


def _build_stop_words():
    words = {entry["vorto"] for entry in _load_vortoj() if entry.get("ignorinda", True)}

    for prefix in CORRELATIVE_PREFIXES:
        for suffix in CORRELATIVE_SUFFIXES:
            words.update(_inflections(prefix + suffix))

    for pronoun in PRONOUNS:
        words.add(pronoun + "n")                      # accusative: min, lin...
        words.update(_inflections(pronoun + "a"))     # possessives: mia, miajn...

    return frozenset(words)


STOP_WORDS = _build_stop_words()
"""Default stop-word set (lowercase)."""


def is_stop_word(word, stop_words=STOP_WORDS):
    """Tells whether a word (in any casing) is a stop-word."""
    return word.lower() in stop_words
