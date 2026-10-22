"""Language object: runs the tokenizer and the component pipeline."""

from .morphology import analyze
from .orthography import from_x_system
from .stopwords import CORRELATIVE_PREFIXES, STOP_WORDS
from .tokenizer import Tokenizer
from .tokens import Doc, MorphAnalysis

# Correlatives ending in -u (kiu, tiu, iu, ĉiu, neniu): pronoun or determiner.
U_CORRELATIVES = {prefix + "u" for prefix in CORRELATIVE_PREFIXES}
NOMINAL_POS = {"NOUN", "PROPN", "ADJ", "NUM", "DET"}


class Tagger:
    """Component that sets `token.pos_` and `token.morph` by rule.

    On top of the ending-based analysis, it applies two context rules:
    - A capitalised word at the start of a sentence is a proper noun if it
      appears as a proper noun elsewhere in the document ("Jesuon ... la Jesuo").
    - A correlative in -u followed by a noun or adjective is a determiner
      ("tiu libro"); otherwise it is a pronoun ("tiu venis").
    """

    def __call__(self, doc):
        analyses = [analyze(token.text, token.is_sent_start) for token in doc]

        proper_nouns = {a.lemma for a in analyses if a.pos == "PROPN"}
        for i, token in enumerate(doc):
            if token.is_sent_start and analyses[i].pos != "PROPN":
                candidate = analyze(token.text, is_sent_start=False)
                if candidate.pos == "PROPN" and candidate.lemma in proper_nouns:
                    analyses[i] = candidate

        for token, analysis in zip(doc, analyses):
            token.pos_ = analysis.pos
            token.morph = MorphAnalysis(analysis.morph)

        for i, analysis in enumerate(analyses):
            if analysis.lemma in U_CORRELATIVES:
                j = i + 1
                if j < len(doc) and doc[j].lower_ == "ĉi":   # tiu ĉi libro
                    j += 1
                if j < len(doc) and doc[j].pos_ in NOMINAL_POS:
                    doc[i].pos_ = "DET"
        return doc


class Lemmatizer:
    """Component that sets `token.lemma_` by rule (strips inflection).

    If the Tagger has already marked the token as a proper noun, the lemma
    keeps its capital letter ("Jesuon" -> "Jesuo"), as it would mid-sentence.
    """

    def __call__(self, doc):
        for token in doc:
            is_sent_start = token.is_sent_start and token.pos_ != "PROPN"
            token.lemma_ = analyze(token.text, is_sent_start).lemma
        return doc


class StopWordsMarker:
    """Component that sets `token.is_stop` from a stop-word list."""

    def __init__(self, stop_words):
        self.stop_words = stop_words

    def __call__(self, doc):
        for token in doc:
            token.is_stop = (
                token.lower_ in self.stop_words
                or token.lemma_.lower() in self.stop_words
            )
        return doc


class Language:
    """Esperanto text processor, analogous to `spacy.Language`.

    Example:
        >>> nlp = Language()
        >>> doc = nlp("La hundo kuras.")
        >>> [token.text for token in doc]
        ['La', 'hundo', 'kuras', '.']
    """

    lang = "eo"

    def __init__(self, stop_words=None, normalize_x=False):
        """
        Args:
            stop_words: set of stop-words; defaults to `STOP_WORDS`.
            normalize_x: convert the x-system ('cx' -> 'ĉ') before analysis.
        """
        self.tokenizer = Tokenizer()
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)
        self.normalize_x = normalize_x
        self.pipeline = [
            ("tagger", Tagger()),
            ("lemmatizer", Lemmatizer()),
            ("stopwords", StopWordsMarker(self.stop_words)),
        ]

    @property
    def pipe_names(self):
        return [name for name, _ in self.pipeline]

    def add_pipe(self, component, name, before=None):
        """Adds a component (a Doc -> Doc function) to the pipeline."""
        position = self.pipe_names.index(before) if before else len(self.pipeline)
        self.pipeline.insert(position, (name, component))

    def make_doc(self, text):
        """Tokenizes the text without running the rest of the pipeline."""
        if self.normalize_x:
            text = from_x_system(text)
        tokens = self.tokenizer(text)
        words = [word for word, _, _ in tokens]
        return Doc(
            text,
            words,
            [idx for _, idx, _ in tokens],
            [ws for _, _, ws in tokens],
            self.tokenizer.sentence_starts(words),
        )

    def __call__(self, text):
        doc = self.make_doc(text)
        for _, component in self.pipeline:
            doc = component(doc)
        return doc

    def pipe(self, texts):
        """Processes a sequence of texts lazily."""
        for text in texts:
            yield self(text)


def load(**kwargs):
    """Creates an Esperanto processor: `nlp = esperanto_nlp.load()`."""
    return Language(**kwargs)
