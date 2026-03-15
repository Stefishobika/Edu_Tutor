import os
import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
chunks_path = os.path.join(BASE_DIR, "..", "data", "processed", "chunks.json")
vectorstore_dir = os.path.join(BASE_DIR, "..", "vectorstore")
vectorizer_path = os.path.join(vectorstore_dir, "tfidf_vectorizer.pkl")
matrix_path = os.path.join(vectorstore_dir, "tfidf_matrix.pkl")

# Create vectorstore folder if not exists
os.makedirs(vectorstore_dir, exist_ok=True)

# Load chunks
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_texts = [chunk["text"] for chunk in chunks]

def build_index():
    """
    Build TF-IDF vectorizer and matrix from all chunks.
    Save them to vectorstore/ for reuse.
    """
    print("🔄 Building TF-IDF index...")

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print("✅ TF-IDF index built and saved.")

def load_index():
    """
    Load previously saved vectorizer + matrix.
    If not found, build automatically.
    """
    if not os.path.exists(vectorizer_path) or not os.path.exists(matrix_path):
        build_index()

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(matrix_path, "rb") as f:
        tfidf_matrix = pickle.load(f)

    return vectorizer, tfidf_matrix

def retrieve(query, top_k=3):
    """
    Retrieve top-k most relevant chunks from the full textbook.
    """
    vectorizer, tfidf_matrix = load_index()

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(similarities[idx]),
            "chunk_id": chunks[idx]["chunk_id"],
            "chapter_title": chunks[idx]["chapter_title"],
            "page_start": chunks[idx]["page_start"],
            "page_end": chunks[idx]["page_end"],
            "text": chunks[idx]["text"]
        })

    return results

if __name__ == "__main__":
    print("\n📘 EduTutor Baseline Retriever")
    print("Type your question below.\n")

    query = input("❓ Enter your question: ").strip()

    if not query:
        print("⚠️ No question entered.")
    else:
        results = retrieve(query, top_k=3)

        print("\n🔍 Top 3 Retrieved Chunks:\n")

        for i, res in enumerate(results, start=1):
            print("=" * 100)
            print(f"Result #{i}")
            print(f"Score      : {res['score']:.4f}")
            print(f"Chunk ID   : {res['chunk_id']}")
            print(f"Chapter    : {res['chapter_title']}")
            print(f"Pages      : {res['page_start']} - {res['page_end']}")
            print(f"Preview    : {res['text'][:500]}...")
            print()