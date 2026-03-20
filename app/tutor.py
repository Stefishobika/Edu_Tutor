"""
tutor.py
--------
Main entry point for EduTutor.
Ties together retrieval + context pruning + LLM answering.

Usage:
    python app/tutor.py
"""

import sys
from retriever import retrieve, build_prompt
from groq import Groq


def get_llm():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    def groq_answer(prompt: str) -> str:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response.choices[0].message.content
    return groq_answer


def run():
    llm = get_llm()

    print("\n" + "=" * 60)
    print("  EduTutor — Personalized AI Tutor")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = retrieve(query)
        cost   = result["cost"]

        print(f"\n  [Cost] baseline {cost['baseline_tokens']} tokens → "
              f"pruned {cost['pruned_tokens']} tokens "
              f"({cost['token_saving_pct']}% saved) | "
              f"retrieval {result['latency_ms']} ms")

        if not result["pruned"]:
            print("\n  No relevant content found in the textbooks for this question.\n")
            continue

        chapters = list({c["chapter_title"] for c in result["pruned"]})
        print(f"  [Source] {', '.join(chapters)}\n")

        prompt = build_prompt(query, result["pruned"])
        answer = llm(prompt)

        print("\nAnswer:")
        print(answer)
        print()


if __name__ == "__main__":
    run()