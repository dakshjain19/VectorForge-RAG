#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o /tmp/vectorforge-db -pthread
SERVER_HOST=127.0.0.1 OLLAMA_HOST=127.0.0.1 /tmp/vectorforge-db > /tmp/vectorforge-test.log 2>&1 &
PID=$!
cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  rm -f /tmp/vectorforge-db /tmp/vectorforge-test.log /tmp/vectorforge-status.json /tmp/vectorforge-invalid.json /tmp/vectorforge-origin.json
}
trap cleanup EXIT

wait_for_server() {
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8080/status >/tmp/vectorforge-status.json; then return 0; fi
    sleep 1
  done
  cat /tmp/vectorforge-test.log
  return 1
}
wait_for_server
grep -q '"demoDims":16' /tmp/vectorforge-status.json

Q='0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1'
curl -fsS "http://127.0.0.1:8080/search?v=$Q&k=5&metric=euclidean&algo=bruteforce" | grep -q '"results":'
curl -fsS "http://127.0.0.1:8080/search?v=$Q&k=5&metric=cosine&algo=kdtree" | grep -q '"results":'

CODE="$(curl -sS -o /tmp/vectorforge-invalid.json -w '%{http_code}' -X POST http://127.0.0.1:8080/insert -H 'Content-Type: application/json' --data '{"metadata":"bad","category":"test","embedding":[1,2]}')"
test "$CODE" = "400"
! curl -fsS http://127.0.0.1:8080/items | grep -q '"metadata":"bad"'

CODE="$(curl -sS -o /tmp/vectorforge-origin.json -w '%{http_code}' -H 'Origin: https://attacker.example' http://127.0.0.1:8080/status)"
test "$CODE" = "403"

# Insert a valid vector, restart, and confirm the persisted record is restored.
curl -fsS -X POST http://127.0.0.1:8080/insert -H 'Content-Type: application/json' --data '{"metadata":"persistence-smoke-test","category":"test","embedding":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]}' >/dev/null
kill "$PID"
wait "$PID" 2>/dev/null || true
SERVER_HOST=127.0.0.1 OLLAMA_HOST=127.0.0.1 /tmp/vectorforge-db >> /tmp/vectorforge-test.log 2>&1 &
PID=$!
wait_for_server
curl -fsS http://127.0.0.1:8080/items | grep -q 'persistence-smoke-test'

echo "VectorForge-RAG build and API smoke tests passed."
