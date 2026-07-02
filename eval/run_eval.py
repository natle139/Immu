"""
Runs eval/golden.json against the live retriever + chain and scores:
- retrieval hit: expected source file was among the retrieved sources
  (or, for out-of-scope questions, nothing was retrieved)
- answer hit: one of the expected keyword phrases appears in the answer

Usage: python -m eval.run_eval
"""
import json
import sys
from pathlib import Path

from src.retriever import build_retriever, retrieve_documents
from src.chain import generate_answer

GOLDEN_PATH = Path(__file__).parent / "golden.json"
PASS_THRESHOLD = 0.7


def check_retrieval(item: dict, sources: list[str]) -> bool:
    expected = item["expected_source"]
    if expected is None:
        return not sources
    expected = [expected] if isinstance(expected, str) else expected
    return any(e in s for e in expected for s in sources)


def check_answer(item: dict, answer: str) -> bool:
    answer_lower = answer.lower().replace("’", "'")
    return any(kw.lower().replace("’", "'") in answer_lower for kw in item["expected_keywords"])


def main():
    golden = json.loads(GOLDEN_PATH.read_text())
    vectorstore = build_retriever()

    retrieval_hits = 0
    answer_hits = 0

    for item in golden:
        documents, sources = retrieve_documents(item["question"], vectorstore)
        answer = generate_answer(item["question"], documents, sources)

        retrieval_ok = check_retrieval(item, sources)
        answer_ok = check_answer(item, answer)
        retrieval_hits += retrieval_ok
        answer_hits += answer_ok

        status = "PASS" if retrieval_ok and answer_ok else "FAIL"
        print(f"[{status}] {item['question']}")
        if not retrieval_ok:
            print(f"         retrieval: expected {item['expected_source']!r}, got {sources}")
        if not answer_ok:
            print(f"         answer: none of {item['expected_keywords']} found in: {answer[:200]}")

    n = len(golden)
    retrieval_score = retrieval_hits / n
    answer_score = answer_hits / n
    print(f"\nRetrieval score: {retrieval_hits}/{n} ({retrieval_score:.0%})")
    print(f"Answer score:    {answer_hits}/{n} ({answer_score:.0%})")

    if retrieval_score < PASS_THRESHOLD or answer_score < PASS_THRESHOLD:
        sys.exit(1)


if __name__ == "__main__":
    main()
