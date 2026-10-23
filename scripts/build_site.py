"""Builds the static web demo (GitHub Pages) into an output folder.

Trains the notebook's final model (TF-IDF + logistic regression, every
article weighted equally) on the whole dataset, exports its weights to
model.json, checks that the pure-Python classifier in web/demo.py gives the
same probabilities as scikit-learn, and packs the esperanto_nlp library so
Pyodide can load it in the browser.

Usage:
    pip install -e . pandas scikit-learn
    python scripts/build_site.py --output _site
    python -m http.server --directory _site   # preview at http://localhost:8000
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
import demo  # noqa: E402  (web/demo.py, shared with the browser)

CLASSES = ["kristanismo", "islamo", "judismo"]
SEED = 42
DECIMALS = 5


def load_dataset():
    df = pd.read_csv(ROOT / "data" / "dataset.csv", sep=";", encoding="utf-8")
    sources = pd.read_csv(ROOT / "data" / "sources.csv", sep=";", encoding="utf-8")
    df["article"] = np.repeat(sources["title"].to_numpy(), sources["paragraphs"].to_numpy())
    df["lemmas"] = [demo.content_lemmas(demo.nlp(text)) for text in df["text"]]
    return df


def train(df):
    """Same pipeline and hyperparameters as the notebook's final model."""
    vectorizer = TfidfVectorizer(analyzer=lambda lemmas: lemmas + [
        " ".join(pair) for pair in zip(lemmas, lemmas[1:])
    ], min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(df["lemmas"])
    weights = 1 / df.groupby("article")["article"].transform("size")
    classifier = LogisticRegression(C=3, max_iter=2000, class_weight="balanced")
    classifier.fit(X, df["class"], sample_weight=(weights / weights.mean()).to_numpy())
    return vectorizer, classifier


def export(vectorizer, classifier):
    order = [list(classifier.classes_).index(c) for c in CLASSES]
    return {
        "classes": CLASSES,
        "terms": vectorizer.get_feature_names_out().tolist(),
        "idf": np.round(vectorizer.idf_, DECIMALS).tolist(),
        "coef": np.round(classifier.coef_[order], DECIMALS).tolist(),
        "intercept": np.round(classifier.intercept_[order], DECIMALS).tolist(),
    }


def check(df, vectorizer, classifier, data, n=300):
    """The browser model must reproduce scikit-learn's probabilities."""
    model = demo.Model(data)
    sample = df.sample(n, random_state=SEED)["lemmas"]
    order = [list(classifier.classes_).index(c) for c in CLASSES]
    expected = classifier.predict_proba(vectorizer.transform(sample))[:, order]
    actual = np.array([model.predict_proba(lemmas) for lemmas in sample])
    error = np.abs(expected - actual).max()
    assert error < 1e-3, f"web model differs from scikit-learn by {error:.2e}"
    print(f"Web model matches scikit-learn (max difference {error:.1e} on {n} paragraphs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()

    df = load_dataset()
    vectorizer, classifier = train(df)
    data = export(vectorizer, classifier)
    check(df, vectorizer, classifier, data)

    out = args.output
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out, ignore=shutil.ignore_patterns("__pycache__"))
    with open(out / "model.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    with zipfile.ZipFile(out / "esperanto_nlp.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "src" / "esperanto_nlp").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(ROOT / "src"))
    size = (out / "model.json").stat().st_size / 1e6
    print(f"{len(data['terms'])} features, model.json {size:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
