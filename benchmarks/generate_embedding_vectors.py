#!/usr/bin/env python3
"""Create a reproducible ANN corpus using real Ollama text embeddings.

The text records are generated from a deterministic template corpus; vectors are
produced by the configured embedding model (not random/synthetic coordinates).
"""
import argparse
import json
import struct
import urllib.request

TOPICS = [
    ("vector indexing", "graph-based indexing, neighbor links, insertion order, and index maintenance"),
    ("database persistence", "binary snapshots, atomic replacement, recovery, and durable storage"),
    ("semantic retrieval", "query embeddings, nearest-neighbor ranking, and relevant context selection"),
    ("distributed systems", "replication, consistency, partitioning, and failure recovery"),
    ("API engineering", "HTTP routes, request validation, status codes, and service contracts"),
    ("machine learning", "feature vectors, model inference, evaluation, and data preprocessing"),
    ("network security", "authentication, transport encryption, input validation, and access control"),
    ("observability", "structured logs, metrics, tracing, latency percentiles, and alerting"),
    ("compiler design", "lexing, parsing, optimization, code generation, and runtime behavior"),
    ("operating systems", "process scheduling, virtual memory, synchronization, and file systems"),
    ("cloud infrastructure", "containers, orchestration, autoscaling, and resource isolation"),
    ("data engineering", "batch ingestion, schema validation, transformations, and lineage"),
    ("search ranking", "candidate generation, relevance scoring, ranking signals, and top-k results"),
    ("graph algorithms", "vertices, edges, traversal, shortest paths, and connectivity"),
    ("concurrency", "thread pools, locks, atomic operations, contention, and race conditions"),
    ("storage engines", "write-ahead logs, page layouts, compaction, and indexing"),
    ("recommendation systems", "user-item interactions, candidate retrieval, ranking, and feedback"),
    ("natural language processing", "tokenization, embeddings, semantic similarity, and language models"),
    ("testing strategy", "unit tests, integration tests, regression suites, and deterministic fixtures"),
    ("performance engineering", "throughput, memory use, cache locality, profiling, and benchmarking"),
]
OPERATIONS = [
    "design a robust implementation for a latency-sensitive service",
    "compare two implementation strategies under realistic workloads",
    "diagnose a production incident and identify its root cause",
    "measure correctness and performance with a repeatable experiment",
    "document trade-offs for maintainability and operational reliability",
    "plan a migration while preserving backward compatibility",
    "validate edge cases and failure behavior before deployment",
    "optimize resource usage without changing externally visible behavior",
    "define service-level objectives and monitor regressions",
    "review architecture choices for scalability and fault tolerance",
]
CONTEXTS = [
    "The workload contains mixed read and write traffic with bursty arrivals.",
    "The implementation must handle malformed input and partial failures safely.",
    "The service runs in a container with limited CPU and memory resources.",
    "The system needs predictable behavior across restarts and software upgrades.",
    "The dataset contains repeated records, skewed distributions, and outliers.",
    "The design should expose useful diagnostics without leaking sensitive data.",
    "The team needs reproducible tests that run in continuous integration.",
    "The system must preserve correctness while reducing tail latency.",
    "The deployment uses local development first and cloud hosting later.",
    "The solution should be measurable rather than relying on anecdotal results.",
]

def make_documents(count):
    docs = []
    for i in range(count):
        topic, details = TOPICS[i % len(TOPICS)]
        operation = OPERATIONS[(i // len(TOPICS) + i) % len(OPERATIONS)]
        context = CONTEXTS[(i * 7 + i // len(TOPICS)) % len(CONTEXTS)]
        docs.append(
            f"Technical record {i:05d}. Subject: {topic}. "
            f"Key concepts: {details}. Engineering task: {operation}. "
            f"Operational context: {context} "
            f"Record variant {i:05d} is included to create a distinct retrieval item."
        )
    return docs

def make_queries(count):
    queries = []
    for i in range(count):
        topic, details = TOPICS[(i * 3) % len(TOPICS)]
        operation = OPERATIONS[(i // len(TOPICS) + i) % len(OPERATIONS)]
        queries.append(
            f"What should an engineer consider about {topic} when they need to "
            f"{operation}? Include {details}."
        )
    return queries

def embed_batch(base_url, model, inputs):
    payload = json.dumps({"model": model, "input": inputs}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode())
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        raise RuntimeError("Ollama /api/embed returned an unexpected embedding count")
    return vectors

def embed_all(base_url, model, texts, batch_size):
    output = []
    for start in range(0, len(texts), batch_size):
        output.extend(embed_batch(base_url, model, texts[start:start + batch_size]))
        print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--documents", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="ann-vectors.bin")
    args = parser.parse_args()

    docs = embed_all(args.ollama_url, args.model, make_documents(args.documents), args.batch_size)
    queries = embed_all(args.ollama_url, args.model, make_queries(args.queries), args.batch_size)
    dims = len(docs[0])
    if dims == 0 or any(len(v) != dims for v in docs + queries):
        raise RuntimeError("Embedding dimensions are empty or inconsistent")

    with open(args.output, "wb") as output:
        output.write(struct.pack("<III", len(docs), len(queries), dims))
        for vector in docs + queries:
            output.write(struct.pack(f"<{dims}f", *vector))
    print(f"Wrote {len(docs)} document vectors and {len(queries)} query vectors ({dims}D) to {args.output}")

if __name__ == "__main__":
    main()
