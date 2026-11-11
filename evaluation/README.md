# Evaluation assets

- `rag_eval.json`: curated regression corpus of **26 technical reference documents and 52 questions** (two differently phrased questions per reference document).
- `rag_eval.py`: inserts the corpus through the REST API, computes Hit Rate@K, MRR@K and Context Precision@K over every question, samples LLM answer generation by default, and removes inserted chunks.
- `../benchmarks/generate_embedding_vectors.py`: creates a deterministic technical text corpus and obtains actual dense vectors from the configured Ollama embedding model for ANN benchmarking.

Run the RAG evaluator against a running VectorForge-RAG server with Ollama and both configured models available:

```bash
python3 evaluation/rag_eval.py --base-url http://127.0.0.1:8080
```

By default, retrieval is measured on all 52 questions while answer generation is run on 10 evenly spaced questions to control runtime. Use `--all-generation` to generate answers for every question, or `--skip-generation` for retrieval-only evaluation. This is a broader curated regression set, not a comprehensive independent benchmark or human evaluation.

Generate the ANN benchmark corpus and run the C++ benchmark:

```bash
ollama pull nomic-embed-text
python3 benchmarks/generate_embedding_vectors.py --documents 10000 --queries 100 --output ann-vectors.bin
g++ -std=c++17 -O2 -Wall -Wextra -pthread benchmarks/ann_benchmark.cpp -o ann-benchmark
./ann-benchmark ann-vectors.bin
```

The corpus records are deterministic templated technical text; vector coordinates are actual Ollama model embeddings (768 dimensions for `nomic-embed-text`). The corpus is not a customer dataset, so its benchmark results should not be represented as production workload performance.
