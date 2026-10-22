"""Builds the Esperanto paragraph dataset from Esperanto Wikipedia.

For each class, the script walks the articles of a Wikipedia category (and its
subcategories down to a given depth), downloads the plain text of each
article, splits it into paragraphs, and each paragraph inherits the article's
class.

Outputs:
    data/dataset.csv   Data contract: UTF-8, ';' separator, text;class columns
    data/sources.csv   Provenance: article, class, revision and paragraph count

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max-articles 100 --depth 0
"""

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# Dataset class -> Esperanto Wikipedia category.
# This is the only place to change to switch topics.
CATEGORIES = {
    "kristanismo": "Kristanismo",
    "islamo": "Islamo",
    "judismo": "Judismo",
}

API_URL = "https://eo.wikipedia.org/w/api.php"
USER_AGENT = "esperanto-text-classification/0.2 (https://github.com/Godalberto10/Esperanto-Text-Classification)"

# Sections that carry no topical text (bibliography, links...).
EXCLUDED_SECTIONS = {
    "vidu ankaŭ", "referencoj", "notoj", "piednotoj", "notoj kaj referencoj",
    "eksteraj ligiloj", "eksteraj ligoj", "ligiloj", "bibliografio",
    "literaturo", "fontoj",
}

# Shorter paragraphs are usually captions, list items or headings.
MIN_WORDS = 15

HEADING_RE = re.compile(r"^(=+)\s*(.*?)\s*=+$")


class WikiClient:
    """Minimal MediaWiki API client with throttling and retries."""

    def __init__(self, delay=0.5, retries=5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.retries = retries

    def get(self, **params):
        params.update(format="json", formatversion=2)
        for attempt in range(self.retries):
            time.sleep(self.delay)
            try:
                response = self.session.get(API_URL, params=params, timeout=30)
                if response.status_code == 429:
                    # Too many requests: wait longer each time (backoff).
                    wait = 10 * (attempt + 1)
                    print(f"  Got 429, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                print(f"  Network error ({error}), retrying...", file=sys.stderr)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API did not respond after {self.retries} attempts: {params}")


def get_category_articles(client, category, depth):
    """Returns the article titles of a category and its subcategories."""
    articles = set()
    visited = set()
    frontier = [f"Kategorio:{category}"]

    for _ in range(depth + 1):
        next_frontier = []
        for cat in frontier:
            if cat in visited:
                continue
            visited.add(cat)
            params = {
                "action": "query", "list": "categorymembers", "cmtitle": cat,
                "cmtype": "page|subcat", "cmlimit": 500,
            }
            while True:
                data = client.get(**params)
                for member in data["query"]["categorymembers"]:
                    if member["ns"] == 0:
                        articles.add(member["title"])
                    elif member["ns"] == 14:
                        next_frontier.append(member["title"])
                if "continue" not in data:
                    break
                params.update(data["continue"])
        frontier = next_frontier

    return articles


def get_article(client, title):
    """Returns (plain text, revision id) of an article, or None on failure."""
    data = client.get(
        action="query", prop="extracts|info", explaintext=1, titles=title,
    )
    page = data["query"]["pages"][0]
    text = page.get("extract")
    if not text:
        return None
    return text, page.get("lastrevid")


def split_paragraphs(text, min_words=MIN_WORDS):
    """Splits an article's text into clean paragraphs.

    Skips headings, the sections in EXCLUDED_SECTIONS (with all their
    subsections) and paragraphs shorter than `min_words` words.
    """
    paragraphs = []
    skip_level = None  # level of the excluded heading we are inside

    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and heading.group(2).lower() in EXCLUDED_SECTIONS:
                skip_level = level
            continue

        if skip_level is None and len(line.split()) >= min_words:
            paragraphs.append(line)

    return paragraphs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=1,
                        help="subcategory depth to walk (default 1)")
    parser.add_argument("--max-articles", type=int, default=200,
                        help="maximum articles per class (default 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    client = WikiClient()
    rng = random.Random(args.seed)

    # 1. Article titles per class.
    titles_by_class = {}
    for label, category in CATEGORIES.items():
        print(f"Walking Kategorio:{category}...")
        titles_by_class[label] = get_category_articles(client, category, args.depth)
        print(f"  {len(titles_by_class[label])} articles")

    # An article in several categories would have an ambiguous label: drop it.
    counts = Counter(t for titles in titles_by_class.values() for t in titles)
    ambiguous = {t for t, n in counts.items() if n > 1}
    print(f"Dropped {len(ambiguous)} articles found in more than one class")

    # 2. Download and split into paragraphs (reproducible random sample).
    rows = []
    sources = []
    for label, titles in titles_by_class.items():
        candidates = sorted(titles - ambiguous)
        rng.shuffle(candidates)
        selected = candidates[:args.max_articles]
        print(f"Downloading {len(selected)} articles for '{label}'...")

        for i, title in enumerate(selected, 1):
            try:
                article = get_article(client, title)
            except RuntimeError as error:
                print(f"  Skipped '{title}': {error}", file=sys.stderr)
                continue
            if article is None:
                continue
            text, revid = article
            paragraphs = split_paragraphs(text)
            rows.extend((p, label) for p in paragraphs)
            sources.append((title, label, revid, len(paragraphs)))
            if i % 25 == 0:
                print(f"  {i}/{len(selected)}")

    # 3. Repeated paragraphs (templates, text copied between articles).
    #    If they repeat with different classes the label is ambiguous: drop them all.
    labels_by_text = {}
    for text, label in rows:
        labels_by_text.setdefault(text, set()).add(label)
    rows = [
        (text, labels.pop())
        for text, labels in labels_by_text.items()
        if len(labels) == 1
    ]

    # 4. Write following the data contract.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "dataset.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["text", "class"])
        writer.writerows(rows)

    with open(args.output_dir / "sources.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["title", "class", "revid", "paragraphs"])
        writer.writerows(sorted(sources))

    print("\nParagraphs per class:")
    for label, n in sorted(Counter(label for _, label in rows).items()):
        print(f"  {label:15} {n}")
    print(f"Total: {len(rows)} -> {args.output_dir / 'dataset.csv'}")


if __name__ == "__main__":
    main()
