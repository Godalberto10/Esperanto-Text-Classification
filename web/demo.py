"""Classifier behind the web demo, run in the browser with Pyodide.

It reproduces the notebook's TF-IDF + logistic regression model in pure
Python, so the browser only needs `esperanto_nlp` and the exported weights
(no NumPy or scikit-learn). `scripts/build_site.py` uses the same
`content_lemmas` to train the model and checks that `Model.predict_proba`
matches scikit-learn.
"""

import json
import math
from collections import Counter

import esperanto_nlp

nlp = esperanto_nlp.load(normalize_x=True)


def content_lemmas(doc):
    """Lowercased content lemmas of a Doc, without stop-words or numbers."""
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 1
    ]


class Model:
    """TF-IDF (unigrams and bigrams, sublinear tf) + multinomial logistic regression."""

    def __init__(self, data):
        self.classes = data["classes"]
        self.terms = data["terms"]
        self.index = {term: i for i, term in enumerate(self.terms)}
        self.idf = data["idf"]
        self.coef = data["coef"]            # one weight list per class
        self.intercept = data["intercept"]

    def features(self, lemmas):
        """Sparse L2-normalised TF-IDF vector as {feature index: value}."""
        ngrams = lemmas + [" ".join(pair) for pair in zip(lemmas, lemmas[1:])]
        counts = Counter(self.index[g] for g in ngrams if g in self.index)
        vector = {j: (1 + math.log(n)) * self.idf[j] for j, n in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {j: v / norm for j, v in vector.items()}

    def scores(self, vector):
        return [
            b + sum(w[j] * v for j, v in vector.items())
            for w, b in zip(self.coef, self.intercept)
        ]

    def predict_proba(self, lemmas):
        scores = self.scores(self.features(lemmas))
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        return [e / sum(exps) for e in exps]

    def contributions(self, vector, k, limit=8):
        """Features that push class k up the most, relative to the other classes."""
        others = [c for c in range(len(self.classes)) if c != k]
        weight = {
            j: v * (self.coef[k][j] - sum(self.coef[c][j] for c in others) / len(others))
            for j, v in vector.items()
        }
        best = sorted(weight, key=weight.get, reverse=True)[:limit]
        return [(self.terms[j], weight[j]) for j in best if weight[j] > 0]


def analyze(text, model):
    """Everything the page shows for a text, as a JSON string."""
    doc = nlp(text)
    lemmas = content_lemmas(doc)
    vector = model.features(lemmas)
    proba = model.predict_proba(lemmas)
    best = max(range(len(proba)), key=proba.__getitem__)
    return json.dumps({
        "probabilities": dict(zip(model.classes, proba)),
        "predicted": model.classes[best],
        "known_features": len(vector),
        "evidence": model.contributions(vector, best),
        "tokens": [
            {"text": t.text, "lemma": t.lemma_, "pos": t.pos_,
             "morph": str(t.morph), "stop": t.is_stop}
            for t in doc
        ],
    }, ensure_ascii=False)
