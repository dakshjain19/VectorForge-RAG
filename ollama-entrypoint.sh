#!/bin/sh
ollama serve &
echo "Waiting for Ollama to start..."
until ollama list > /dev/null 2>&1; do
  sleep 2
done
echo "Ollama ready, pulling models..."
ollama pull nomic-embed-text
ollama pull llama3.2
echo "Models ready!"
wait