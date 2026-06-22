"""
corpus/scraper.py
Scrapes PSA grading guides and Wikipedia sports card pages.
Saves raw documents as .txt files in corpus/raw/
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project; contact: maheshpooja261@gmail.com)"
}

# ── PSA Grading Guide URLs ──────────────────────────────────────────────────
PSA_URLS = [
    {
        "url": "https://www.psacard.com/resources/gradingstandards",
        "filename": "psa_grading_standards.txt",
        "label": "PSA Grading Standards"
    },
    {
        "url": "https://www.psacard.com/resources/gradingstandards/sportsCards",
        "filename": "psa_sports_card_grading.txt",
        "label": "PSA Sports Card Grading"
    },
    {
        "url": "https://www.psacard.com/resources/gradingstandards/tradingCards",
        "filename": "psa_trading_card_grading.txt",
        "label": "PSA Trading Card Grading"
    },
]

# ── Wikipedia URLs ───────────────────────────────────────────────────────────
WIKIPEDIA_URLS = [
    {
        "url": "https://en.wikipedia.org/wiki/Baseball_card",
        "filename": "wiki_baseball_card.txt",
        "label": "Wikipedia: Baseball Card"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Basketball_card",
        "filename": "wiki_basketball_card.txt",
        "label": "Wikipedia: Basketball Card"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Football_card",
        "filename": "wiki_football_card.txt",
        "label": "Wikipedia: Football Card"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Sports_card_collecting",
        "filename": "wiki_sports_card_collecting.txt",
        "label": "Wikipedia: Sports Card Collecting"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Professional_Sports_Authenticator",
        "filename": "wiki_psa.txt",
        "label": "Wikipedia: Professional Sports Authenticator"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Beckett_Grading_Services",
        "filename": "wiki_bgs.txt",
        "label": "Wikipedia: Beckett Grading Services"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Rookie_card",
        "filename": "wiki_rookie_card.txt",
        "label": "Wikipedia: Rookie Card"
    },
    {
        "url": "https://en.wikipedia.org/wiki/1952_Topps",
        "filename": "wiki_1952_topps.txt",
        "label": "Wikipedia: 1952 Topps Set"
    },
    {
        "url": "https://en.wikipedia.org/wiki/1986%E2%80%9387_Fleer_Basketball",
        "filename": "wiki_1986_fleer_basketball.txt",
        "label": "Wikipedia: 1986-87 Fleer Basketball"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Upper_Deck_Company",
        "filename": "wiki_upper_deck.txt",
        "label": "Wikipedia: Upper Deck"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Topps",
        "filename": "wiki_topps.txt",
        "label": "Wikipedia: Topps"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Panini_Group",
        "filename": "wiki_panini.txt",
        "label": "Wikipedia: Panini Group"
    },
]


def clean_text(soup: BeautifulSoup) -> str:
    """
    Remove navigation, footers, scripts, and styles.
    Return clean body text with section headings preserved.
    """
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Preserve heading structure as plain text markers
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        level = int(heading.name[1])
        prefix = "#" * level + " "
        heading.string = prefix + heading.get_text(strip=True)

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remove blank lines
    cleaned = "\n".join(lines)

    return cleaned


def scrape_page(url: str, label: str) -> str | None:
    """Fetch a single page and return cleaned text, or None on failure."""
    try:
        print(f"  Fetching: {label}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = clean_text(soup)
        print(f"  OK — {len(text):,} chars")
        return text
    except requests.exceptions.RequestException as e:
        print(f"  FAILED — {e}")
        return None


def save_document(text: str, filename: str, source_url: str, label: str):
    """Save text with metadata header to raw/ directory."""
    filepath = RAW_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: {source_url}\n")
        f.write(f"LABEL: {label}\n")
        f.write(f"CHARS: {len(text)}\n")
        f.write("---\n")
        f.write(text)
    print(f"  Saved → {filepath}")


def run():
    print("=" * 60)
    print("SpinWheel Card Intelligence — Corpus Scraper")
    print("=" * 60)

    all_sources = [
        ("PSA Grading Guides", PSA_URLS),
        ("Wikipedia Articles", WIKIPEDIA_URLS),
    ]

    total_saved = 0
    total_failed = 0

    for section_name, urls in all_sources:
        print(f"\n── {section_name} ──")
        for item in urls:
            text = scrape_page(item["url"], item["label"])
            if text and len(text) > 500:  # skip pages with almost no content
                save_document(text, item["filename"], item["url"], item["label"])
                total_saved += 1
            else:
                print(f"  SKIPPED — too little content or fetch failed")
                total_failed += 1
            time.sleep(1)  # polite delay between requests

    print("\n" + "=" * 60)
    print(f"Done. Saved: {total_saved} | Failed/Skipped: {total_failed}")
    print(f"Raw documents in: {RAW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
