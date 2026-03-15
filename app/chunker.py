import os
import json
import re

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
pages_path = os.path.join(BASE_DIR, "..", "data", "processed", "pages.json")
chapters_path = os.path.join(BASE_DIR, "..", "data", "processed", "chapters_final.json")
output_path = os.path.join(BASE_DIR, "..", "data", "processed", "chunks.json")

# Settings
CHUNK_SIZE_WORDS = 350   # ideal size for hackathon demo
OVERLAP_WORDS = 50       # overlap between chunks

# Load data
with open(pages_path, "r", encoding="utf-8") as f:
    pages_data = json.load(f)

with open(chapters_path, "r", encoding="utf-8") as f:
    chapters_data = json.load(f)

# Convert pages list to dict for fast lookup
pages_dict = {page["page_number"]: page["text"] for page in pages_data}

def clean_text(text):
    """Basic cleaning to normalize whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(words, chunk_size=350, overlap=50):
    """Split word list into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(chunk_words)

        if end == len(words):
            break

        start += (chunk_size - overlap)

    return chunks

all_chunks = []

for chapter in chapters_data:
    chapter_title = chapter["chapter_title"]
    start_page = chapter["start_page"]
    end_page = chapter["end_page"]

    # Collect all text for this chapter
    chapter_pages = []
    for page_num in range(start_page, end_page + 1):
        if page_num in pages_dict:
            chapter_pages.append(pages_dict[page_num])

    chapter_text = "\n".join(chapter_pages)
    chapter_text = clean_text(chapter_text)

    if not chapter_text:
        continue

    words = chapter_text.split()
    chunked = chunk_text(words, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

    for idx, chunk_words in enumerate(chunked, start=1):
        chunk_text_str = " ".join(chunk_words)

        chunk_obj = {
            "chunk_id": f"{chapter_title.lower().replace(' ', '_')}_{idx}",
            "chapter_title": chapter_title,
            "page_start": start_page,
            "page_end": end_page,
            "chunk_index": idx,
            "text": chunk_text_str
        }

        all_chunks.append(chunk_obj)

# Save chunks
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)

print(f"\n✅ chunks.json created at:\n{output_path}")
print(f"📦 Total chunks created: {len(all_chunks)}\n")

# Show sample preview
print("Sample chunks:\n")
for chunk in all_chunks[:5]:
    print("=" * 80)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Chapter: {chunk['chapter_title']}")
    print(f"Pages: {chunk['page_start']} - {chunk['page_end']}")
    print(f"Preview: {chunk['text'][:250]}...")
    print()