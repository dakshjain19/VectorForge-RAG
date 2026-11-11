#!/usr/bin/env python3
"""Evaluate VectorForge retrieval on a larger curated corpus and sample answer generation."""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "rag_eval.json"


def request_json(base, path, method="GET", payload=None, timeout=240):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path, data=body, method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path}: HTTP {exc.code}: {detail}") from exc


def choose_generation_indices(total, sample_count):
    if sample_count >= total:
        return set(range(total))
    if sample_count == 1:
        return {0}
    return {round(i * (total - 1) / (sample_count - 1)) for i in range(sample_count)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generation-sample", type=int, default=10,
                        help="number of evenly spaced queries used for LLM answer diagnostics")
    parser.add_argument("--all-generation", action="store_true",
                        help="generate answers for every query (slower)")
    args = parser.parse_args()
    if args.generation_sample < 1:
        parser.error("--generation-sample must be at least 1")

    golden = json.loads(DATA.read_text(encoding="utf-8"))
    inserted_ids = []
    hits, reciprocal_ranks, precisions = [], [], []
    keyword_scores, generation_seconds = [], []
    generation_indices = (
        set() if args.skip_generation else
        (set(range(len(golden["queries"])) if args.all_generation else
         choose_generation_indices(len(golden["queries"]), args.generation_sample)))
    )

    status = request_json(args.base_url, "/status")
    if not status.get("ollamaAvailable"):
        raise RuntimeError("Ollama is not available according to /status")

    try:
        for doc in golden["documents"]:
            result = request_json(args.base_url, "/doc/insert", "POST", doc)
            ids = result.get("ids", [])
            if not ids:
                raise RuntimeError(f"Document insertion returned no IDs: {doc['title']}")
            inserted_ids.extend(ids)

        for index, case in enumerate(golden["queries"]):
            retrieved = request_json(
                args.base_url, "/doc/search", "POST",
                {"question": case["question"], "k": args.k},
            )
            contexts = retrieved.get("contexts", [])
            titles = [item.get("title", "") for item in contexts]
            relevant = case["relevant_title"]
            rank = next((i + 1 for i, title in enumerate(titles)
                         if title == relevant or title.startswith(relevant + " [")), None)
            hits.append(1 if rank else 0)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            precisions.append(sum(
                1 for title in titles
                if title == relevant or title.startswith(relevant + " [")
            ) / max(1, len(titles)))

            if index in generation_indices:
                started = time.perf_counter()
                answer_result = request_json(
                    args.base_url, "/doc/ask", "POST",
                    {"question": case["question"], "k": args.k}, timeout=300,
                )
                generation_seconds.append(time.perf_counter() - started)
                answer = answer_result.get("answer", "").lower()
                keywords = [word.lower() for word in case.get("answer_keywords", [])]
                keyword_scores.append(
                    sum(1 for word in keywords if word in answer) / max(1, len(keywords))
                )
            print(f"{'PASS' if rank else 'MISS'} | {case['question']} | rank={rank} | retrieved={titles}")

        report = {
            "document_count": len(golden["documents"]),
            "query_count": len(golden["queries"]),
            "k": args.k,
            "hit_rate_at_k": sum(hits) / max(1, len(hits)),
            "mrr_at_k": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
            "context_precision_at_k": sum(precisions) / max(1, len(precisions)),
            "generation_query_count": len(generation_indices),
            "answer_keyword_coverage": (
                sum(keyword_scores) / len(keyword_scores) if keyword_scores else None
            ),
            "mean_generation_seconds": (
                sum(generation_seconds) / len(generation_seconds)
                if generation_seconds else None
            ),
            "generation_evaluated": bool(generation_indices),
        }
        print("\nRAG EVALUATION")
        print(json.dumps(report, indent=2))
        if report["hit_rate_at_k"] < 0.8:
            raise RuntimeError("Retrieval hit-rate gate failed (expected >= 0.80)")
    finally:
        for doc_id in reversed(inserted_ids):
            try:
                request_json(args.base_url, f"/doc/delete/{doc_id}", "DELETE", timeout=30)
            except Exception as exc:
                print(f"WARNING: could not clean up evaluation document {doc_id}: {exc}",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
