"""
Runs eval/golden.json against the live retriever + chain and scores:
- retrieval hit: expected source file was among the retrieved sources
  (or, for out-of-scope questions, nothing was retrieved)
- answer hit: one of expected_patterns matches the normalised answer and
  none of must_not_contain does (all are regexes, see normalise())

Usage: python -m eval.run_eval [--runs 3] [--out eval/results/baseline.json]
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.retriever import CONFIDENCE_THRESHOLD, TOP_K, build_retriever, retrieve_documents
from src.chain import MODEL, TEMPERATURE, generate_answer

GOLDEN_PATH = Path(__file__).parent / "golden.json"
PASS_THRESHOLD = 0.7


def normalise(text: str) -> str:
    """Lowercase, straighten quotes/hyphens, drop markdown, collapse spaces so patterns match meaning, not formatting."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[‐-―]", "-", text)  # non-breaking / en / em dashes the model emits
    text = re.sub(r"[*_`#>]", "", text)
    return re.sub(r"[ \t ]+", " ", text)


def check_retrieval(item: dict, sources: list[str]) -> bool:
    if "expected_sources_all" in item:  # comparison questions need every visa's fact sheet
        return all(any(e in s for s in sources) for e in item["expected_sources_all"])
    expected = item["expected_source"]
    if expected is None:
        return not sources
    expected = [expected] if isinstance(expected, str) else expected
    return any(e in s for e in expected for s in sources)


def check_answer(item: dict, answer: str) -> bool:
    text = normalise(answer)
    if any(re.search(p, text) for p in item.get("must_not_contain", [])):
        return False
    return any(re.search(p, text) for p in item["expected_patterns"])


def run_once(golden: list[dict], vectorstore) -> list[dict]:
    results = []
    for i, item in enumerate(golden, 1):
        documents, sources = retrieve_documents(item["question"], vectorstore)
        answer = generate_answer(item["question"], documents, sources)
        r = dict(q=i, question=item["question"], retrieval_ok=check_retrieval(item, sources),
                 answer_ok=check_answer(item, answer), sources=sources, answer=answer)
        results.append(r)

        print(f"[{'PASS' if r['retrieval_ok'] and r['answer_ok'] else 'FAIL'}] Q{i} {item['question']}")
        if not r["retrieval_ok"]:
            print(f"         retrieval: got {sources}")
        if not r["answer_ok"]:
            print(f"         answer: {answer[:200]!r}")
    return results


def git_info() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text())
    vectorstore = build_retriever()
    n = len(golden)

    runs = []
    for run in range(args.runs):
        print(f"\n=== run {run + 1}/{args.runs}")
        results = run_once(golden, vectorstore)
        retrieval = sum(r["retrieval_ok"] for r in results) / n
        answer = sum(r["answer_ok"] for r in results) / n
        print(f"Retrieval {retrieval:.0%}  Answer {answer:.0%}")
        runs.append({"retrieval": retrieval, "answer": answer, "results": results})

    summary = {k: {"mean": mean(r[k] for r in runs), "min": min(r[k] for r in runs), "max": max(r[k] for r in runs)}
               for k in ("retrieval", "answer")}
    print(f"\nRetrieval mean {summary['retrieval']['mean']:.0%} (min {summary['retrieval']['min']:.0%}, max {summary['retrieval']['max']:.0%})")
    print(f"Answer    mean {summary['answer']['mean']:.0%} (min {summary['answer']['min']:.0%}, max {summary['answer']['max']:.0%})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_info(),
            "config": {"model": MODEL, "temperature": TEMPERATURE, "k": TOP_K, "confidence_threshold": CONFIDENCE_THRESHOLD},
            "questions": n,
            "runs": args.runs,
            "summary": summary,
            "per_run": runs,
        }, indent=1))
        print(f"saved {args.out}")

    if summary["retrieval"]["mean"] < PASS_THRESHOLD or summary["answer"]["mean"] < PASS_THRESHOLD:
        sys.exit(1)


if __name__ == "__main__":
    main()
