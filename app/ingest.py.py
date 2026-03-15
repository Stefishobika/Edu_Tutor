import fitz
import os
import json

# Base folder of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
pdf_path = os.path.join(BASE_DIR, "..", "data", "textbooks", "Class_10_Science.pdf")
output_path = os.path.join(BASE_DIR, "..", "data", "processed", "pages.json")

# Open PDF
doc = fitz.open(pdf_path)

pages_data = []

print(f"Total pages found: {len(doc)}\n")

for i in range(len(doc)):
    try:
        page = doc[i]
        text = page.get_text("text").strip()   # explicitly ask for plain text

        pages_data.append({
            "page_number": i + 1,
            "text": text
        })

        # show progress every 25 pages
        if (i + 1) % 25 == 0 or (i + 1) == len(doc):
            print(f"Processed page {i + 1}/{len(doc)}")

    except Exception as e:
        print(f"⚠️ Error on page {i + 1}: {e}")
        pages_data.append({
            "page_number": i + 1,
            "text": ""
        })

# Save JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pages_data, f, ensure_ascii=False, indent=4)

print(f"\n✅ Page-wise text saved successfully to:\n{output_path}")
print(f"Total pages saved: {len(pages_data)}")