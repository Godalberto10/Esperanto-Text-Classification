"""Doc, Token and Span objects that structure the analysis output."""

import unicodedata

NUMBER_WORDS = {
    "nul", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ",
    "dek", "cent", "mil", "miliono", "miliardo",
}


class MorphAnalysis(dict):
    """Morphological features of a token, e.g. {"Number": "Plur", "Case": "Acc"}.

    Rendered in Universal Dependencies format: "Case=Acc|Number=Plur".
    """

    def to_dict(self):
        return dict(self)

    def __str__(self):
        return "|".join(f"{key}={value}" for key, value in sorted(self.items()))


class Token:
    """A word, number or punctuation mark inside a Doc."""

    def __init__(self, doc, i, text, idx, whitespace="", is_sent_start=False):
        self.doc = doc
        self.i = i                      # token index in the Doc
        self.text = text
        self.idx = idx                  # character offset in the text
        self.whitespace_ = whitespace   # whitespace following the token
        self.is_sent_start = is_sent_start
        # Attributes filled in by the pipeline components.
        self.lemma_ = ""
        self.pos_ = ""
        self.morph = MorphAnalysis()
        self.is_stop = False

    @property
    def text_with_ws(self):
        return self.text + self.whitespace_

    @property
    def lower_(self):
        return self.text.lower()

    @property
    def is_alpha(self):
        """True if it is a word (letters, with inner hyphens or apostrophes)."""
        stripped = self.text.replace("-", "").replace("'", "").replace("’", "")
        return stripped.isalpha()

    @property
    def is_digit(self):
        return self.text.isdigit()

    @property
    def like_num(self):
        """True if it represents a number: '1200', '3,14', '20-a', 'dek'."""
        return self.text[:1].isdigit() or self.lower_ in NUMBER_WORDS

    @property
    def is_punct(self):
        return all(unicodedata.category(c).startswith("P") for c in self.text)

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Span:
    """Contiguous sequence of tokens in a Doc (e.g. a sentence)."""

    def __init__(self, doc, start, end):
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def text(self):
        if self.start >= self.end:
            return ""
        first, last = self.doc[self.start], self.doc[self.end - 1]
        return self.doc.text[first.idx:last.idx + len(last)]

    def __iter__(self):
        return iter(self.doc.tokens[self.start:self.end])

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        return list(self)[i]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Doc:
    """Result of analysing a text: a sequence of Tokens."""

    def __init__(self, text, words, starts, whitespaces, sent_starts):
        self.text = text
        self.tokens = [
            Token(self, i, word, idx, ws, sent_start)
            for i, (word, idx, ws, sent_start)
            in enumerate(zip(words, starts, whitespaces, sent_starts))
        ]

    @property
    def sents(self):
        """Yields the document's sentences as Span objects."""
        start = 0
        for token in self.tokens[1:]:
            if token.is_sent_start:
                yield Span(self, start, token.i)
                start = token.i
        if self.tokens:
            yield Span(self, start, len(self.tokens))

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end, step = key.indices(len(self.tokens))
            if step != 1:
                raise ValueError("Spans only support a step of 1")
            return Span(self, start, end)
        return self.tokens[key]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text
