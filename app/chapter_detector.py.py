import os
import json
import re

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
pages_path = os.path.join(BASE_DIR, "..", "data", "processed", "pages.json")
chapters_output_path = os.path.join(BASE_DIR, "..", "data", "processed", "chapters_raw.json")

# Load page-wise text
with open(pages_path, "r", encoding="utf-8") as f:
    pages_data = json.load(f)

chapter_candidates = []

# Patterns to detect chapter headings
chapter_patterns = [
    r"\bChapter\s+\d+\b",   # Chapter 1, Chapter 2
    r"\bCHAPTER\s+\d+\b",
    r"^\d+\.\s+[A-Z][A-Za-z ,\-()]+",   # 1. Laws of Motion
]

for page in pages_data:
    page_number = page["page_number"]
    text = page["text"].strip()

    if not text:
        continue

    # Look only at first 20 lines of each page (chapter titles usually appear near top)
    lines = text.split("\n")[:20]
    page_top = "\n".join(lines)

    matched = False
    for pattern in chapter_patterns:
        if re.search(pattern, page_top, re.MULTILINE):
            chapter_candidates.append({
                "page_number": page_number,
                "preview": page_top[:500]
            })
            matched = True
            break

# Save raw candidates
with open(chapters_output_path, "w", encoding="utf-8") as f:
    json.dump(chapter_candidates, f, ensure_ascii=False, indent=4)

print(f"\n✅ Found {len(chapter_candidates)} possible chapter-start pages.")
print(f"Saved raw candidates to:\n{chapters_output_path}\n")

# Print them in terminal for inspection
for c in chapter_candidates:
    print("=" * 80)
    print(f"PAGE {c['page_number']}")
    print("-" * 80)
    print(c["preview"])
    print()