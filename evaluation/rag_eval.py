#!/usr/bin/env python3
"""Evaluate VectorForge-RAG retrieval and answer behavior against a tiny golden set.

Uses only the Python standard library. The server and Ollama must already be running.
"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--skip-generation", action="store_true",
                        help="measure retrieval only; do not call the LLM")
    args = parser.parse_args()
    golden = json.loads(DATA.read_text(encoding="utf-8"))
    inserted_ids = []
    hits = []
    reciprocal_ranks = []
    precisions = []
    keyword_scores = []
    generation_seconds = []

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

        for case in golden["queries"]:
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

            if not args.skip_generation:
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
            "answer_keyword_coverage": (
                sum(keyword_scores) / len(keyword_scores) if keyword_scores else None
            ),
            "mean_generation_seconds": (
                sum(generation_seconds) / len(generation_seconds)
                if generation_seconds else None
            ),
            "generation_evaluated": not args.skip_generation,
        }
        print("\nRAG EVALUATION")
        print(json.dumps(report, indent=2))
        # Retrieval is the deterministic gate. Answer keyword coverage is reported
        # as a diagnostic because local model output can vary across versions.
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
