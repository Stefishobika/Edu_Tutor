"""
retriever.py
------------
Loads the pre-built TF-IDF index and retrieves the most
relevant chunks for a given query, with context pruning
and chapter title boosting.
"""

import os
import json
import pickle
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH     = os.path.join(BASE_DIR, "..", "data", "processed", "chunks.json")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "vectorstore", "tfidf_vectorizer.pkl")
MATRIX_PATH     = os.path.join(BASE_DIR, "..", "vectorstore", "tfidf_matrix.pkl")

MIN_SCORE         = 0.10
MAX_CONTEXT_WORDS = 600
PRUNED_TOP_K      = 5
CHAPTER_BOOST     = 0.15

TOPIC_MAP = {
    "Laws of motion": [
        "inertia", "newton", "force", "momentum", "friction",
        "gravity", "acceleration", "velocity", "motion", "thrust",
        "gravitation", "torque", "resultant", "equilibrium"
    ],
    "Optics": [
        "light", "lens", "mirror", "reflection", "refraction",
        "focal", "concave", "convex", "image", "eye", "spectrum",
        "scattering", "dispersion", "magnification"
    ],
    "Thermal Physics": [
        "heat", "temperature", "thermal", "expansion", "gas",
        "boyle", "charles", "pressure", "conduction", "convection",
        "radiation", "evaporation", "specific"
    ],
    "Electricity": [
        "current", "voltage", "resistance", "ohm", "circuit",
        "conductor", "insulator", "power", "charge", "electric",
        "battery", "resistor", "parallel", "series"
    ],
    "Acoustics": [
        "sound", "wave", "frequency", "amplitude", "pitch",
        "echo", "resonance", "ultrasound", "noise", "decibel"
    ],
    "Nuclear Physics": [
        "nuclear", "atom", "radioactive", "fission", "fusion",
        "proton", "neutron", "isotope", "radiation", "decay"
    ],
    "Atoms and Molecules": [
        "atom", "molecule", "element", "compound", "avogadro",
        "mole", "atomic", "mass", "formula", "valency"
    ],
    "Periodic Classification of Elements": [
        "periodic", "element", "group", "period", "metal",
        "nonmetal", "valence", "atomic number", "mendeleev"
    ],
    "Solutions": [
        "solution", "solute", "solvent", "concentration",
        "saturated", "osmosis", "diffusion", "suspension", "colloid"
    ],
    "Types of Chemical Reactions": [
        "chemical", "reaction", "oxidation", "reduction",
        "acid", "base", "salt", "neutralization", "combustion",
        "displacement", "decomposition", "combination"
    ],
    "Carbon and its Compounds": [
        "carbon", "organic", "hydrocarbon", "alkane", "alkene",
        "alcohol", "functional", "isomer", "covalent", "allotrope"
    ],
    "Plant Anatomy and Plant Physiology": [
        "photosynthesis", "chlorophyll", "stomata", "leaf",
        "root", "stem", "tissue", "meristem", "xylem", "phloem",
        "plant", "anatomy", "physiology", "transpiration"
    ],
    "Transportation in Plants and Circulation in Animals": [
        "xylem", "phloem", "transpiration", "osmosis", "diffusion",
        "translocation", "circulation", "blood", "heart", "vessel",
        "transport", "ascent", "sap", "plasmolysis"
    ],
    "Nervous System": [
        "neuron", "brain", "nerve", "reflex", "synapse",
        "nervous", "impulse", "spinal", "receptor", "effector"
    ],
    "Reproduction in Plants": [
        "pollination", "fertilization", "seed", "fruit",
        "vegetative", "spore", "germination", "flower", "reproduction"
    ],
    "Reproduction in Animals": [
        "reproduction", "embryo", "fertilization", "zygote",
        "puberty", "menstrual", "ovum", "sperm", "sexual", "asexual"
    ],
    "Heredity": [
        "heredity", "gene", "chromosome", "dna", "trait",
        "dominant", "recessive", "mendel", "allele", "mutation"
    ],
    "Structural Organisation of Animals": [
        "tissue", "organ", "epithelial", "muscle", "connective",
        "nervous tissue", "organisation", "system"
    ],
    "Relations and Functions": [
        "relation", "function", "domain", "range", "mapping",
        "cartesian", "ordered", "pair", "composition", "inverse"
    ],
    "Numbers and Sequences": [
        "arithmetic", "geometric", "progression", "sequence",
        "series", "euclid", "modular", "fibonacci", "hcf", "lcm"
    ],
    "Algebra": [
        "quadratic", "polynomial", "matrix", "equation",
        "factorization", "simultaneous", "linear", "gcd", "lcm",
        "rational", "expression", "determinant"
    ],
    "Geometry": [
        "triangle", "similarity", "congruent", "thales",
        "pythagoras", "circle", "tangent", "concurrency", "angle",
        "bisector", "midpoint"
    ],
    "Coordinate Geometry": [
        "coordinate", "slope", "line", "area", "quadrilateral",
        "distance", "midpoint", "gradient", "intercept", "collinear"
    ],
    "Trigonometry": [
        "sine", "cosine", "tangent", "trigonometry", "angle",
        "identity", "height", "distance", "elevation", "depression"
    ],
    "Mensuration": [
        "area", "volume", "surface", "cylinder", "cone",
        "sphere", "frustum", "perimeter", "mensuration", "solid"
    ],
    "Statistics and Probability": [
        "mean", "median", "mode", "probability", "statistics",
        "variance", "deviation", "event", "sample", "frequency"
    ],
}


def _load_chunks():
    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_PATH}.\n"
            "Run chunker.py first: python app/chunker.py"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"chunks.json is corrupted: {e}")


def _load_index():
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError("TF-IDF index not found. Run chunker.py first.")

    chunks_mtime = os.path.getmtime(CHUNKS_PATH)
    index_mtime  = os.path.getmtime(VECTORIZER_PATH)
    if chunks_mtime > index_mtime:
        print("[warn] chunks.json is newer than the index. Run chunker.py again.")

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        matrix = pickle.load(f)
    return vectorizer, matrix


def _chapter_boost(query: str, chapter_title: str) -> float:
    stopwords = {"what", "is", "are", "the", "a", "an", "of",
                 "in", "and", "how", "does", "do", "explain",
                 "define", "describe", "give", "write", "about"}
    query_words = {w.lower() for w in query.split() if w.lower() not in stopwords}

    # Direct word overlap with chapter title
    chapter_words = {w.lower() for w in chapter_title.split()}
    if query_words & chapter_words:
        return CHAPTER_BOOST

    # Topic keyword mapping
    keywords = TOPIC_MAP.get(chapter_title, [])
    if query_words & set(keywords):
        return CHAPTER_BOOST

    return 0.0


def _prune(candidates):
    pruned     = []
    word_count = 0
    for chunk in candidates:
        if chunk["score"] < MIN_SCORE:
            break
        words = len(chunk["text"].split())
        if word_count + words > MAX_CONTEXT_WORDS:
            continue
        pruned.append(chunk)
        word_count += words
    return pruned


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


def cost_report(baseline_chunks, pruned_chunks):
    baseline_text = " ".join(c["text"] for c in baseline_chunks)
    pruned_text   = " ".join(c["text"] for c in pruned_chunks)
    b_tokens = estimate_tokens(baseline_text)
    p_tokens = estimate_tokens(pruned_text)
    saving   = round((1 - p_tokens / b_tokens) * 100, 1) if b_tokens else 0
    return {
        "baseline_chunks":  len(baseline_chunks),
        "pruned_chunks":    len(pruned_chunks),
        "baseline_tokens":  b_tokens,
        "pruned_tokens":    p_tokens,
        "token_saving_pct": saving,
    }


def retrieve(query: str, top_k: int = PRUNED_TOP_K) -> dict:
    t0 = time.time()

    chunks             = _load_chunks()
    vectorizer, matrix = _load_index()

    query_vec    = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()

    boosted = np.array([
        similarities[i] + _chapter_boost(query, chunks[i]["chapter_title"])
        for i in range(len(chunks))
    ])

    top_indices = np.argsort(boosted)[::-1][:top_k]
    baseline    = []
    for idx in top_indices:
        baseline.append({
            "score":         float(similarities[idx]),
            "boosted_score": float(boosted[idx]),
            "chunk_id":      chunks[idx]["chunk_id"],
            "chapter_title": chunks[idx]["chapter_title"],
            "page_start":    chunks[idx]["page_start"],
            "page_end":      chunks[idx]["page_end"],
            "text":          chunks[idx]["text"],
        })

    pruned     = _prune(baseline)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "pruned":     pruned,
        "baseline":   baseline,
        "cost":       cost_report(baseline, pruned),
        "latency_ms": latency_ms,
    }


def build_prompt(query: str, pruned_chunks) -> str:
    if not pruned_chunks:
        context = "No relevant content found in the textbook for this question."
    else:
        context_parts = []
        for i, chunk in enumerate(pruned_chunks, 1):
            context_parts.append(
                f"[Source {i} — {chunk['chapter_title']}, "
                f"pages {chunk['page_start']}–{chunk['page_end']}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

    return (
        "You are a helpful tutor for Indian state-board students. "
        "Answer the question using ONLY the textbook excerpts below. "
        "Be concise, clear, and curriculum-aligned. "
        "If the answer is not in the excerpts, say so honestly.\n\n"
        f"Textbook excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


if __name__ == "__main__":
    print("\n📘 EduTutor Retriever  (with context pruning + chapter boost)\n")
    query = input("Enter your question: ").strip()
    if not query:
        print("No question entered.")
    else:
        result = retrieve(query)
        c = result["cost"]
        print(f"\n  Baseline: {c['baseline_tokens']} tokens → "
              f"Pruned: {c['pruned_tokens']} tokens "
              f"({c['token_saving_pct']}% saved) | {result['latency_ms']} ms")
        for i, ch in enumerate(result["pruned"], 1):
            print(f"\n  Result #{i}  score={ch['score']:.4f}  chapter={ch['chapter_title']}")
            print(f"  Preview: {ch['text'][:300]}...")