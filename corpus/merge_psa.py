"""
corpus/merge_psa.py
Deduplicates the three identical PSA scraped files into one,
then merges with the manual grades 1-9 reference.
Run this once, then delete the three originals.
"""

from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

# The three scraped files (identical content)
PSA_SCRAPED = [
    RAW_DIR / "psa_grading_standards.txt",
    RAW_DIR / "psa_sports_card_grading.txt",
    RAW_DIR / "psa_trading_card_grading.txt",
]

# Grades 1-9 that were behind JavaScript — not in the scrape
GRADES_1_TO_9 = """
# PSA Grades 1 through 9 — Complete Definitions

## PSA 9 — Mint
A PSA Mint 9 is a superb condition card that exhibits only one of the following minor flaws: a very slight wax stain on the reverse, a minor printing imperfection or slightly out-of-register print on a non-crucial area. The card must have 60/40 or better centering in both directions on the front of the card, and 90/10 or better centering on the reverse. Four sharp corners, sharp focus and full original gloss are also required. The card must be free of stains or creases.

## PSA 8 — Near Mint to Mint
A PSA Near Mint to Mint 8 card shows only slight fraying on one or two corners. There may be slight scratching visible only upon close inspection. The surface must have original gloss with a minor printing imperfection allowed. The card must have 65/35 or better centering in both directions on the front, and 90/10 or better centering on the reverse. No creases are allowed.

## PSA 7 — Near Mint
A PSA Near Mint 7 card is a well-centered card with only minor wear visible on the card's corners and/or edges. There may be slight scuffing or light scratches visible upon close inspection. The card must have 70/30 or better centering in both directions on the front, and 90/10 or better centering on the reverse. The gloss may be slightly broken. No creases are allowed.

## PSA 6 — Excellent to Mint
A PSA Excellent to Mint 6 card shows slight surface wear beginning to show on the card's corners and/or edges. There may be slight scuffing or light scratches. The card must have 80/20 or better centering in both directions on the front. The surface may show some loss of original gloss. No creases are allowed.

## PSA 5 — Excellent
A PSA Excellent 5 card has outstanding eye appeal. A card in this grade shows moderate wear on the card's corners and/or edges. There may be minor scuffing or light scratches visible upon close inspection. The card must have 85/15 or better centering in both directions on the front. The surface may show some loss of original gloss. No creases are allowed.

## PSA 4 — Very Good to Excellent
A PSA Very Good to Excellent 4 card has significant wear on corners and edges. There may be several creases and notable loss of original gloss. Centering must be 85/15 or better on the front.

## PSA 3 — Very Good
A PSA Very Good 3 card shows heavy wear on the card's corners and edges, with possible light creasing. There may be heavy scuffing, light staining, and moderate loss of original gloss. Centering must be 85/15 or better on the front.

## PSA 2 — Good
A PSA Good 2 card shows heavy wear and some creasing. The card may show heavy loss of gloss, possible staining, possible tears, or extreme wear. Centering must be 85/15 or better on the front.

## PSA 1 — Poor
A PSA Poor 1 card displays heavy creasing, possible tape marks, writing on the front or back, severe staining, torn borders, or missing pieces of the card. The card is identifiable beyond question.

## PSA 1.5 — Fair
A PSA Fair 1.5 card shows excessive wear on its corners and surface. A PSA Fair 1.5 card may display heavy creasing or crinkle marks throughout the card as well as possible tape marks or writing. Though a 1.5 is heavily worn, the card itself is still intact. The image on the card is still identifiable.
"""


def extract_body(filepath: Path) -> str:
    """Extract body text after the --- separator."""
    content = filepath.read_text(encoding="utf-8")
    parts = content.split("---\n", 1)
    return parts[1] if len(parts) > 1 else content


def clean_navigation(text: str) -> str:
    """Remove slide navigation artifacts from the scraped content."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove known navigation artifacts
        if stripped in ("Previous slide", "Next slide", "-", "PSA | Grading Standards"):
            continue
        # Remove the duplicate menu items at the top
        if stripped in ("Card Grading", "Ticket Grading", "Pack Grading",
                        "Comic Book and Magazine Grading", "Video Game Grading",
                        "Grade Definitions", "GEM-MT"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def run():
    print("Merging PSA files...")

    # All three scraped files are identical — just use the first one
    source_file = PSA_SCRAPED[0]
    if not source_file.exists():
        print(f"ERROR: {source_file} not found. Run scraper.py first.")
        return

    scraped_body = extract_body(source_file)
    scraped_clean = clean_navigation(scraped_body)

    # Build merged document
    merged = (
        "SOURCE: https://www.psacard.com/resources/gradingstandards\n"
        "LABEL: PSA Complete Grading Reference (scraped + supplemented)\n"
        "CHARS: merged\n"
        "---\n"
        "# PSA Complete Grading Standards Reference\n\n"
        "## Overview\n"
        "Professional Sports Authenticator (PSA) is the world's largest and most trusted "
        "third-party trading card authentication and grading company. PSA uses a numeric "
        "grading scale from 1 to 10, where 10 represents a perfect card and 1 represents "
        "a heavily damaged card.\n\n"
        + scraped_clean
        + "\n\n"
        + GRADES_1_TO_9
    )

    output_path = RAW_DIR / "psa_complete_reference.txt"
    output_path.write_text(merged, encoding="utf-8")
    print(f"Saved merged file → {output_path}")
    print(f"Total chars: {len(merged):,}")

    # Delete the three duplicate originals
    for f in PSA_SCRAPED:
        if f.exists():
            f.unlink()
            print(f"Deleted duplicate: {f.name}")

    print("\nDone. Run chunker.py next.")


if __name__ == "__main__":
    run()
