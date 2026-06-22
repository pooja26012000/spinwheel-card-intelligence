"""
corpus/chunker.py
Splits raw .txt documents into semantically meaningful chunks.
Saves chunks as JSONL in corpus/chunks/
"""

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
CHUNKS_DIR = Path(__file__).parent / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

# Chunking configuration — document these for interviews
CHUNK_SIZE = 512        # target tokens per chunk (~4 chars per token → ~2048 chars)
CHUNK_OVERLAP = 50      # overlap in tokens to preserve context across chunk boundaries
MIN_CHUNK_CHARS = 100   # discard chunks shorter than this — likely navigation remnants
CHARS_PER_TOKEN = 4     # approximation for token estimation without a tokenizer


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def split_by_headings(text: str) -> list[str]:
    """
    Primary strategy: split on markdown-style headings (# ## ###)
    preserved during scraping. This is semantic splitting — each
    section becomes its own candidate chunk.
    """
    heading_pattern = re.compile(r"(?=^#{1,4} .+$)", re.MULTILINE)
    sections = heading_pattern.split(text)
    return [s.strip() for s in sections if s.strip()]


def split_by_paragraphs(text: str) -> list[str]:
    """
    Fallback strategy: split on double newlines (paragraph breaks)
    used when no heading structure is present.
    """
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def merge_short_sections(sections: list[str], max_chars: int) -> list[str]:
    """
    Merge consecutive short sections until we approach max_chars.
    Prevents the index from being flooded with tiny single-sentence chunks.
    """
    merged = []
    current = ""

    for section in sections:
        if not current:
            current = section
        elif len(current) + len(section) + 1 <= max_chars:
            current = current + "\n" + section
        else:
            merged.append(current)
            current = section

    if current:
        merged.append(current)

    return merged


def split_large_chunk(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    If a single section exceeds max_chars, split it at sentence
    boundaries with overlap to preserve context.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Start next chunk with overlap from end of previous
            overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = (overlap_text + " " + sentence).strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_document(filepath: Path) -> list[dict]:
    """
    Main chunking pipeline for a single document.
    Returns list of chunk dicts with metadata.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Parse metadata header
    lines = content.split("\n")
    source = ""
    label = ""
    text_start = 0

    for i, line in enumerate(lines):
        if line.startswith("SOURCE:"):
            source = line.replace("SOURCE:", "").strip()
        elif line.startswith("LABEL:"):
            label = line.replace("LABEL:", "").strip()
        elif line == "---":
            text_start = i + 1
            break

    body = "\n".join(lines[text_start:])

    max_chars = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN

    # Try heading-based splitting first
    sections = split_by_headings(body)

    # Fall back to paragraph splitting if no headings found
    if len(sections) <= 1:
        sections = split_by_paragraphs(body)

    # Merge sections that are too short
    sections = merge_short_sections(sections, max_chars)

    # Split sections that are too large
    final_sections = []
    for section in sections:
        if len(section) > max_chars:
            final_sections.extend(split_large_chunk(section, max_chars, overlap_chars))
        else:
            final_sections.append(section)

    # Build chunk objects
    chunks = []
    for i, chunk_text in enumerate(final_sections):
        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue  # discard navigation remnants

        chunk = {
            "chunk_id": f"{filepath.stem}_{i:04d}",
            "source_file": filepath.name,
            "source_url": source,
            "label": label,
            "chunk_index": i,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "estimated_tokens": estimate_tokens(chunk_text),
        }
        chunks.append(chunk)

    return chunks


def run():
    print("=" * 60)
    print("SpinWheel Card Intelligence — Semantic Chunker")
    print("=" * 60)
    print(f"\nConfig: chunk_size={CHUNK_SIZE} tokens | overlap={CHUNK_OVERLAP} tokens")
    print(f"Strategy: heading-based → paragraph fallback → sentence split\n")

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        print("No .txt files found in corpus/raw/ — run scraper.py first.")
        return

    all_chunks = []
    output_file = CHUNKS_DIR / "chunks.jsonl"

    for filepath in raw_files:
        chunks = chunk_document(filepath)
        all_chunks.extend(chunks)

        token_counts = [c["estimated_tokens"] for c in chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        print(f"  {filepath.name}")
        print(f"    → {len(chunks)} chunks | avg {avg_tokens:.0f} tokens | "
              f"min {min(token_counts, default=0)} | max {max(token_counts, default=0)}")

    # Write all chunks to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output: {output_file}")
    print(f"{'=' * 60}")

    # Print sample chunk for inspection
    if all_chunks:
        print("\nSample chunk (first chunk from first document):")
        sample = all_chunks[0]
        print(f"  chunk_id: {sample['chunk_id']}")
        print(f"  label: {sample['label']}")
        print(f"  tokens (est): {sample['estimated_tokens']}")
        print(f"  text preview: {sample['text'][:200]}...")


if __name__ == "__main__":
    run()
