"""
chunker.py
----------
Ingests one or more textbook PDFs from data/textbooks/.
Run this ONCE (or whenever you add a new textbook).
"""

import os
import re
import json
import pickle

import fitz
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "..", "data", "textbooks")
PROC_DIR   = os.path.join(BASE_DIR, "..", "data", "processed")
VS_DIR     = os.path.join(BASE_DIR, "..", "vectorstore")

CHUNKS_PATH     = os.path.join(PROC_DIR, "chunks.json")
VECTORIZER_PATH = os.path.join(VS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH     = os.path.join(VS_DIR, "tfidf_matrix.pkl")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(VS_DIR,   exist_ok=True)

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

# Exact chapter names as they appear in the PDF headers
# (case-insensitive matching is used, so casing here doesn't matter)
SCIENCE_CHAPTERS = [
    "Laws of motion",
    "Optics",
    "Thermal Physics",
    "Electricity",
    "Acoustics",
    "Nuclear Physics",
    "Atoms and Molecules",
    "Periodic Classification of Elements",
    "Solutions",
    "Types of Chemical Reactions",
    "Carbon and its Compounds",
    "Plant Anatomy and Plant Physiology",
    "Structural Organisation of Animals",
    "Transportation in Plants and Circulation in Animals",
    "Nervous System",
    "Reproduction in Plants",
    "Reproduction in Animals",
    "Heredity",
    "Origin and Evolution of Life",
    "Environmental Management",
    "Applied Biology",
]

MATHS_CHAPTERS = [
    "Relations and Functions",
    "Numbers and Sequences",
    "Algebra",
    "Geometry",
    "Coordinate Geometry",
    "Trigonometry",
    "Mensuration",
    "Statistics and Probability",
]

TAMIL_CHAPTERS = [
    "தமிழ்மொழி வாழ்த்து",
    "ஆழிக்கு இரண",
    "தமிழ் வரிவடிவ வளர்ச்சி",
    "தொல்காப்பியம்",
    "எழுத்துகளின் பிறப்பு",
    "இயற்கையைப் போற்றுவோம்",
    "பட்டமரம்",
    "இயற்கை தமிழர் மருத்துவம்",
    "வேட்டுக்கிளியும் மருகுமானும்",
    "மயங்கொலிகள்",
    "திருக்குறள்",
    "படறிந்து ஒழுகுதல்",
    "புத்தியைத் தீட்டு",
    "பல்துறைக் கல்வி",
    "ஆன்ம குடிப்பிறத்தல்",
    "விரைமுற்று",
    "திருக்கேதாரம்",
    "குழலினிது யாழினிது",
    "நாட்டுப்புறக் கைவினைக் கலைகள்",
    "தமிழர் இசைக்கருவிகள்",
    "எச்சம்",
    "வளம் பெருகுக",
    "கொங்குநாட்டு வணிகம்",
    "காலம் உடன் வரும்",
    "விரையகால் அமையும் தொடர்கள்",
    "சிங்கி பெற்ற பரிசு",
    "நல்ல நாடு பாரத மாதா",
    "ஒருவன் இருக்கிறான்",
    "இடைச்சொல் உரிச்சொல்",
    "ஒன்றே குலம்",
    "மெய்ஞ்ஞான ஒளி",
    "அயோத்திதாசர் சிந்தனைகள்",
    "மனித யந்திரம்",
    "வேற்றுமை",
    "நோயும் மருந்தும்",
    "குன்றென் நிமிர்ந்துநில்",
    "பால் மனம்",
    "புணர்ச்சி",
]

KNOWN_CHAPTERS = SCIENCE_CHAPTERS + MATHS_CHAPTERS + TAMIL_CHAPTERS

# Skip these generic page headers that are NOT chapter titles
SKIP_LINES = [
    "10th standard science",
    "standard science",
    "government of tamilnadu",
    "science",
    "introduction",
]

HEADING_RE = re.compile(
    r"^(?:chapter|unit|section|lesson|இயல்|பாடம்)\s*\d+",
    re.IGNORECASE,
)


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_num": i, "text": text})
    doc.close()
    return pages


def detect_chapter(text, current_chapter):
    first_lines = [l.strip() for l in text.splitlines()[:10] if l.strip()]

    for line in first_lines:
        # Skip generic headers
        if any(skip in line.lower() for skip in SKIP_LINES):
            continue
        # Match known chapter titles
        for known in KNOWN_CHAPTERS:
            if known.lower() == line.lower() and len(line) < 80:
                return known
            if known.lower() in line.lower() and len(line) < 60:
                return known
        # Generic pattern fallback
        if HEADING_RE.match(line) and len(line) < 80:
            return line

    return current_chapter


def split_into_chunks(words, page_map, chapter, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "text":          " ".join(words[start:end]),
            "chapter_title": chapter,
            "page_start":    page_map[start],
            "page_end":      page_map[end - 1],
        })
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks_from_pdf(pdf_path, subject):
    pages = extract_pages(pdf_path)
    if not pages:
        print(f"  [warn] No text extracted from {pdf_path}")
        return []

    all_chunks  = []
    current_ch  = subject
    word_buffer = []
    page_buffer = []
    chunk_index = 0

    def flush(chapter):
        nonlocal chunk_index
        if not word_buffer:
            return
        for c in split_into_chunks(word_buffer[:], page_buffer[:], chapter, CHUNK_SIZE, CHUNK_OVERLAP):
            slug = re.sub(r"\W+", "_", chapter.lower())[:40]
            c["chunk_id"] = f"{slug}_{chunk_index}"
            chunk_index += 1
            all_chunks.append(c)
        word_buffer.clear()
        page_buffer.clear()

    for page in pages:
        detected = detect_chapter(page["text"], current_ch)
        if detected != current_ch:
            flush(current_ch)
            current_ch = detected
            print(f"    → Chapter: {current_ch}")

        words = page["text"].split()
        word_buffer.extend(words)
        page_buffer.extend([page["page_num"]] * len(words))

    flush(current_ch)
    return all_chunks


def build_tfidf_index(chunks):
    print("\n  Building TF-IDF index...")
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(matrix, f)
    print(f"  TF-IDF index saved  ({matrix.shape[0]} chunks, {matrix.shape[1]} features)")


def ingest_all():
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}.")
        return

    all_chunks = []
    for filename in sorted(pdf_files):
        subject = os.path.splitext(filename)[0].replace("_", " ").title()
        path    = os.path.join(RAW_DIR, filename)
        print(f"\nProcessing: {filename}")
        chunks = build_chunks_from_pdf(path, subject)
        all_chunks.extend(chunks)
        print(f"  {len(chunks)} chunks extracted")

    if not all_chunks:
        print("No chunks produced.")
        return

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_chunks)} total chunks → {CHUNKS_PATH}")

    build_tfidf_index(all_chunks)
    print("\nIngestion complete. You can now run tutor.py.")


if __name__ == "__main__":
    ingest_all()