"""
app.py  —  Flask backend for EduTutor
Run:  python app/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, session
from retriever import retrieve, build_prompt
from groq import Groq

app = Flask(__name__)
app.secret_key = "edututor-secret-key-2024"

GROQ_API_KEY = "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"

def get_answer(query):
    client = Groq(api_key=GROQ_API_KEY)
    result = retrieve(query)
    cost   = result["cost"]

    if not result["pruned"]:
        return {
            "answer":       "Sorry, I couldn't find relevant content in your textbooks for this question. Try rephrasing it.",
            "sources":      [],
            "saving_pct":   cost["token_saving_pct"],
            "baseline_tok": cost["baseline_tokens"],
            "pruned_tok":   cost["pruned_tokens"],
            "latency_ms":   result["latency_ms"],
        }

    prompt = build_prompt(query, result["pruned"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    answer  = response.choices[0].message.content
    sources = list({c["chapter_title"] for c in result["pruned"]})

    return {
        "answer":       answer,
        "sources":      sources,
        "saving_pct":   cost["token_saving_pct"],
        "baseline_tok": cost["baseline_tokens"],
        "pruned_tok":   cost["pruned_tokens"],
        "latency_ms":   result["latency_ms"],
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty question"}), 400
    try:
        result = get_answer(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)