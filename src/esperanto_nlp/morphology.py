"""Rule-based morphological analysis of Esperanto.

Esperanto marks the part of speech with regular endings:

    -o  noun          -a  adjective   -e  adverb
    -i  infinitive    -as/-is/-os present/past/future
    -us conditional   -u  jussive/imperative
    -j  plural        -n  accusative

`analyze` applies these rules to an isolated word and returns its lemma, its
part of speech (Universal Dependencies tags) and its morphological features.
Function words (article, prepositions, pronouns, correlatives...) do not
follow the endings and are resolved with a closed lexicon.

The lemma only strips inflection (number, case, verb tense), not
derivation: "konataj" -> "konata" (not "koni").
"""

import re
import unicodedata
from functools import lru_cache
from typing import NamedTuple

from .stopwords import CORRELATIVE_PREFIXES, FUNCTION_WORDS, PRONOUNS


class Analysis(NamedTuple):
    lemma: str
    pos: str
    morph: dict


# --- Closed lexicon -----------------------------------------------------------

# vortoj.json category -> UD tag. Categories not listed here
# (verboformo, ĝentilaĵo, titolo) are analysed by their ending: "esti", "bona".
CATEGORY_POS = {
    "artikolo": "DET",
    "prepozicio": "ADP",
    "konjunkcio": "CCONJ",
    "pronomo": "PRON",
    "adverbo": "ADV",
    "partikulo": "PART",
    "nombro": "NUM",
    "interjekcio": "INTJ",
}
SUBORDINATING = {"ke", "se", "ĉar", "kvankam", "ol", "kvazaŭ"}

# Correlative ending -> part of speech. Those in -u (kiu, tiu...) can be a
# pronoun or a determiner; the Tagger decides from context.
CORRELATIVE_POS = {
    "u": "PRON", "o": "PRON", "a": "DET", "es": "DET",
    "e": "ADV", "am": "ADV", "al": "ADV", "el": "ADV", "om": "ADV",
}

# Abbreviations: lowercase form -> (lemma, part of speech).
ABBREVIATIONS = {
    "s-ro": ("sinjoro", "NOUN"), "s-ino": ("sinjorino", "NOUN"),
    "d-ro": ("doktoro", "NOUN"), "n-ro": ("numero", "NOUN"),
    "jc.": ("jarcento", "NOUN"), "ĉ.": ("ĉirkaŭ", "ADV"),
    "k.t.p.": ("kaj tiel plu", "ADV"), "ktp.": ("kaj tiel plu", "ADV"),
    "k.s.": ("kaj simile", "ADV"), "t.e.": ("tio estas", "ADV"),
    "ekz.": ("ekzemple", "ADV"),
    "p.k.": ("post Kristo", "ADV"), "a.k.": ("antaŭ Kristo", "ADV"),
}

# Unicode treats these as punctuation, but UD tags them as symbols.
SYMBOLS = {"%", "‰", "&", "@", "#", "*", "/", "§", "†", "‡"}

PLURAL_ACC = {
    "": {"Number": "Sing", "Case": "Nom"},
    "j": {"Number": "Plur", "Case": "Nom"},
    "n": {"Number": "Sing", "Case": "Acc"},
    "jn": {"Number": "Plur", "Case": "Acc"},
}


def _build_lexicon():
    lexicon = {}

    for word, category in FUNCTION_WORDS.items():
        if category in CATEGORY_POS:
            pos = "SCONJ" if word in SUBORDINATING else CATEGORY_POS[category]
            lexicon[word] = Analysis(word, pos, {})

    for prefix in CORRELATIVE_PREFIXES:
        for suffix, pos in CORRELATIVE_POS.items():
            word = prefix + suffix
            lexicon[word] = Analysis(word, pos, {})
            if suffix in ("u", "a"):
                for ending in ("j", "n", "jn"):
                    lexicon[word + ending] = Analysis(word, pos, dict(PLURAL_ACC[ending]))
            elif suffix in ("o", "e"):
                lexicon[word + "n"] = Analysis(word, pos, {"Case": "Acc"})

    for pronoun in PRONOUNS:
        lexicon[pronoun] = Analysis(pronoun, "PRON", {"Case": "Nom"})
        lexicon[pronoun + "n"] = Analysis(pronoun, "PRON", {"Case": "Acc"})
        possessive = pronoun + "a"
        for ending, features in PLURAL_ACC.items():
            lexicon[possessive + ending] = Analysis(possessive, "DET", {"Poss": "Yes", **features})

    lexicon["l'"] = Analysis("la", "DET", {})
    lexicon["adiaŭ"] = Analysis("adiaŭ", "INTJ", {})
    lexicon["ambaŭ"] = Analysis("ambaŭ", "DET", {})
    lexicon["almenaŭ"] = Analysis("almenaŭ", "ADV", {})
    return lexicon


LEXICON = _build_lexicon()


# --- Endings ------------------------------------------------------------------

# (ending, part of speech, lemma ending, features), longest first so that
# "-ojn" is tried before "-o".
ENDINGS = [
    ("ojn", "NOUN", "o", {"Number": "Plur", "Case": "Acc"}),
    ("ajn", "ADJ", "a", {"Number": "Plur", "Case": "Acc"}),
    ("oj", "NOUN", "o", {"Number": "Plur", "Case": "Nom"}),
    ("aj", "ADJ", "a", {"Number": "Plur", "Case": "Nom"}),
    ("on", "NOUN", "o", {"Number": "Sing", "Case": "Acc"}),
    ("an", "ADJ", "a", {"Number": "Sing", "Case": "Acc"}),
    ("en", "ADV", "e", {"Case": "Acc"}),  # direction: hejmen "homewards"
    ("as", "VERB", "i", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Pres"}),
    ("is", "VERB", "i", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"}),
    ("os", "VERB", "i", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Fut"}),
    ("us", "VERB", "i", {"VerbForm": "Fin", "Mood": "Cnd"}),
    ("o", "NOUN", "o", {"Number": "Sing", "Case": "Nom"}),
    ("a", "ADJ", "a", {"Number": "Sing", "Case": "Nom"}),
    ("e", "ADV", "e", {}),
    ("i", "VERB", "i", {"VerbForm": "Inf"}),
    ("u", "VERB", "i", {"VerbForm": "Fin", "Mood": "Jus"}),
]
MIN_ROOT = 2  # avoids treating single-letter fragments as roots

# Participle suffixes: tense and voice.
PARTICIPLES = {
    "ant": ("Pres", "Act"), "int": ("Past", "Act"), "ont": ("Fut", "Act"),
    "at": ("Pres", "Pass"), "it": ("Past", "Pass"), "ot": ("Fut", "Pass"),
}
PARTICIPLE_RE = re.compile(r"^(.{2,}?)(ant|int|ont|at|it|ot)$")

# Roots that end like a participle without being one (ŝtata, milita, spirita...).
NOT_PARTICIPLES = {
    "ŝtat", "milit", "rilat", "spirit", "privat", "konstant", "kvadrat",
    "sacerdot", "jezuit", "kvalit", "majoritat", "eksplicit", "vulgat",
    "golgot", "front", "spit", "debat", "subit", "akurat", "ermit", "edit",
    "kalifat", "sultanat", "patriarkat", "episkopat", "demokrat", "soldat",
    "kandidat", "advokat", "senat", "format", "klimat", "aŭtomat", "strat",
    "prelat", "apostat", "unit", "infinit", "kvant", "eksterordinat",
}


def _participle_features(root):
    match = PARTICIPLE_RE.match(root)
    if not match or root in NOT_PARTICIPLES:
        return {}
    tense, voice = PARTICIPLES[match.group(2)]
    return {"VerbForm": "Part", "Tense": tense, "Voice": voice}


def analyze_by_ending(word):
    """Analyses a lowercase word by its ending, or returns None."""
    for ending, pos, lemma_ending, features in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_ROOT:
            root = word[:-len(ending)]
            morph = dict(features)
            if pos in ("ADJ", "ADV"):
                morph.update(_participle_features(root))
            if pos == "VERB" and root == "est":
                pos = "AUX"
            return Analysis(root + lemma_ending, pos, morph)
    return None


def _proper_noun(text):
    """Proper noun: strips the accusative/plural from Esperantized names."""
    lemma = re.sub(r"-j?n$", "", text)                   # Noa-n -> Noa
    lemma = re.sub(r"(?<=[oa])(jn|j|n)$", "", lemma)     # Jesuon -> Jesuo
    return Analysis(lemma, "PROPN", {})


@lru_cache(maxsize=100_000)
def analyze(text, is_sent_start=False):
    """Returns the `Analysis` (lemma, UD tag, features) of an isolated token.

    Args:
        text: the token as it appears in the text.
        is_sent_start: whether the token starts a sentence (a capital letter
            then does not imply a proper noun).
    """
    if not any(c.isalnum() for c in text):
        is_punct = text not in SYMBOLS and all(
            unicodedata.category(c).startswith("P") for c in text
        )
        return Analysis(text, "PUNCT" if is_punct else "SYM", {})

    if text[0].isdigit():
        if "-" in text:                                   # ordinal: 20-a
            return Analysis(text.lower(), "ADJ", {"NumType": "Ord"})
        return Analysis(text, "NUM", {"NumType": "Card"})

    lower = text.lower()
    if lower in ABBREVIATIONS:
        lemma, pos = ABBREVIATIONS[lower]
        return Analysis(lemma, pos, {"Abbr": "Yes"})
    if lower in LEXICON:
        return LEXICON[lower]

    if text[0].isupper() and not is_sent_start:
        return _proper_noun(text)

    if lower.endswith(("'", "’")):                        # poetic elision: dom' -> domo
        return Analysis(lower[:-1] + "o", "NOUN", {})

    analysis = analyze_by_ending(lower)
    if analysis is not None:
        return analysis
    if text[0].isupper():
        return _proper_noun(text)
    return Analysis(lower, "X", {})


def lemmatize(text, is_sent_start=False):
    """Lemma of an isolated token: lemmatize("estis") -> "esti"."""
    return analyze(text, is_sent_start).lemma


def pos_tag(text, is_sent_start=False):
    """UD tag of an isolated token: pos_tag("hundojn") -> "NOUN"."""
    return analyze(text, is_sent_start).pos
