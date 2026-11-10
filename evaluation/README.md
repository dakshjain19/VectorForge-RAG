# Evaluation assets

- `rag_eval.json`: small golden corpus and question set for retrieval and answer diagnostics.
- `rag_eval.py`: inserts the corpus through the REST API, computes Hit Rate@K, MRR@K and Context Precision@K, optionally evaluates generated answers, then removes inserted chunks.

Run the script against a running VectorForge-RAG server with Ollama and both configured models available:

```bash
python3 evaluation/rag_eval.py --base-url http://127.0.0.1:8080
```

Use `--skip-generation` for retrieval-only evaluation. This is a small regression set, not a comprehensive benchmark or human evaluation.
