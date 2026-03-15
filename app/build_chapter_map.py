import os
import json
import re

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
pages_path = os.path.join(BASE_DIR, "..", "data", "processed", "pages.json")
output_path = os.path.join(BASE_DIR, "..", "data", "processed", "chapters_final.json")

# Known chapter titles from your textbook
KNOWN_CHAPTERS = [
    "Laws of motion",
    "Optics",
    "Thermal Physics",
    "Electricity",
    "Acoustics",
    "Nuclear physics",
    "Atoms and molecules",
    "Periodic Classification of Elements",
    "Solutions",
    "Types of Chemical Reactions",
    "Carbon and its Compounds",
    "Plant Anatomy and Plant Physiology",
    "Structural Organisation of Animals",
    "Transportation in Plants and Circulation in Animals",
    "Nervous System",
    "Plant and Animal Hormones",
    "Reproduction in Plants and Animals",
    "Genetics",
    "Origin and Evolution of Life",
    "Breeding and Biotechnology",
    "Health and Diseases",
    "Environmental Management",
    "Visual Communication",
    "Practicals"
]

# Load page-wise text
with open(pages_path, "r", encoding="utf-8") as f:
    pages_data = json.load(f)

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()

# Store first occurrence of each chapter
chapter_starts = []

for page in pages_data:
    page_num = page["page_number"]
    text = page["text"]

    # Only inspect first 20 lines of each page
    top_text = "\n".join(text.split("\n")[:20])
    norm_top = normalize(top_text)

    for chapter in KNOWN_CHAPTERS:
        if normalize(chapter) in norm_top:
            chapter_starts.append({
                "chapter_title": chapter,
                "page_number": page_num
            })
            break

# Remove duplicates (keep first occurrence only)
seen = set()
unique_starts = []

for item in chapter_starts:
    title = item["chapter_title"]
    if title not in seen:
        seen.add(title)
        unique_starts.append(item)

# Build final chapter ranges
chapters_final = []

for i in range(len(unique_starts)):
    chapter_title = unique_starts[i]["chapter_title"]
    start_page = unique_starts[i]["page_number"]

    if i < len(unique_starts) - 1:
        end_page = unique_starts[i + 1]["page_number"] - 1
    else:
        end_page = pages_data[-1]["page_number"]

    chapters_final.append({
        "chapter_title": chapter_title,
        "start_page": start_page,
        "end_page": end_page
    })

# Save final chapter map
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chapters_final, f, ensure_ascii=False, indent=4)

print(f"\n✅ chapters_final.json created at:\n{output_path}\n")
print("Detected chapter ranges:\n")

for ch in chapters_final:
    print(f"{ch['chapter_title']}: Page {ch['start_page']} to {ch['end_page']}")