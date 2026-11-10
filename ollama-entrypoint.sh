#!/bin/sh
set -eu

ollama serve &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' INT TERM

echo "Waiting for Ollama API..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "Pulling required models (first start can take several minutes)..."
ollama pull nomic-embed-text
ollama pull llama3.2
echo "Required models are ready."
wait "$ollama_pid"
