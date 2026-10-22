"""Regular-expression-based tokenizer."""

import re

# Abbreviations that contain periods and must neither be split nor end a sentence.
ABBREVIATIONS = [
    "k.t.p.", "ktp.", "k.s.", "t.e.", "ekz.",  # etcetera, and similar, that is...
    "p.K.", "a.K.", "a.n.e.", "n.e.",          # AD/BC, (before the) common era
    "jc.", "ĉ.", "vol.", "pp.",                # century, approximately, volume, pages
]

_ABBR = "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True))
_LETTER = r"[^\W\d_]"  # any Unicode letter (including ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)

TOKEN_RE = re.compile(
    rf"""
      (?<!{_LETTER})(?i:{_ABBR})(?!{_LETTER})        # abbreviations: p.K., k.t.p.
    | {_LETTER}+(?:-{_LETTER}+)*(?:['’](?!{_LETTER}))? # words: domo, s-ro, Noa-n, l'
    | \d+(?:[.,]\d+)*(?:-{_LETTER}+)?                 # numbers: 1200, 3,14, 20-a
    | \.\.\.                                          # ellipsis
    | \S                                              # any other symbol
    """,
    re.VERBOSE,
)

SENTENCE_END = {".", "!", "?", "...", "…"}
CLOSING_PUNCT = {"»", "”", '"', "'", ")", "]", "›"}


class Tokenizer:
    """Splits a text into tokens and marks sentence starts."""

    def __call__(self, text):
        """Returns a list of (text, offset, trailing_whitespace) tuples."""
        matches = list(TOKEN_RE.finditer(text))
        tokens = []
        for n, match in enumerate(matches):
            end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
            tokens.append((match.group(), match.start(), text[match.end():end]))
        return tokens

    @staticmethod
    def sentence_starts(words):
        """Tells, for each token, whether it starts a new sentence.

        A sentence starts after a sentence-final mark (. ! ? ...). Closing
        quotes or brackets still belong to the previous sentence, and after
        them a sentence only starts if the next word is capitalised:
        «Ne!» Bone (two sentences) versus «Venu!» li diris (one sentence).
        """
        starts = []
        for i, word in enumerate(words):
            if i == 0:
                starts.append(True)
                continue
            if word in CLOSING_PUNCT:
                starts.append(False)
                continue
            j = i - 1
            while j > 0 and words[j] in CLOSING_PUNCT:
                j -= 1
            after_closing = j < i - 1
            starts.append(
                words[j] in SENTENCE_END
                and (not after_closing or not word[:1].islower())
            )
        return starts
