# VectorForge-RAG — C++ Vector Database & Local RAG

Daksh Jain ([@dakshjain19](https://github.com/dakshjain19)).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first **Vector Database** built from scratch in C++ with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

**Project focus:** This is primarily a backend and retrieval-systems engineering project: indexing, nearest-neighbor search, persistence, REST APIs, local RAG orchestration, and benchmarking. It demonstrates systems skills rather than model training, so it complements a separate forecasting / predictive-ML project instead of replacing one.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (C++)                   ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, often achieving sublinear search behavior in practice; brute force remains an exact O(N·d) baseline. HNSW is approximate, and performance depends on graph parameters and data.

---

## Running with Docker (Recommended)

Docker is the easiest way to run this project — no need to install g++, MSYS2, or manually pull Ollama models.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Project Structure

The Docker setup uses the root-level `Dockerfile`, `docker-compose.yml`, `main.cpp`, `httplib.h`, `index.html`, and `ollama-entrypoint.sh`. The complete repository layout, including evaluation, benchmark, test, and CI folders, is shown in [Project Structure](#project-structure).

### Dockerfile

```dockerfile
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o db -pthread

FROM ubuntu:22.04 AS runner
WORKDIR /app
COPY --from=builder /app/db .
COPY --from=builder /app/index.html .
RUN mkdir -p /app/data
ENV SERVER_HOST=0.0.0.0
EXPOSE 8080
CMD ["./db"]
```

### docker-compose.yml

```yaml
services:
  vectordb:
    build: .
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      SERVER_HOST: 0.0.0.0
      OLLAMA_HOST: ollama
    volumes:
      - vector_data:/app/data
    depends_on:
      ollama:
        condition: service_healthy

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
      - ./ollama-entrypoint.sh:/entrypoint.sh:ro
    entrypoint: ["/bin/sh", "/entrypoint.sh"]
    healthcheck:
      test: ["CMD-SHELL", "ollama list | grep -q nomic-embed-text && ollama list | grep -q llama3.2"]
      interval: 10s
      timeout: 10s
      retries: 60
      start_period: 30s

volumes:
  ollama_data:
  vector_data:
```

### ollama-entrypoint.sh

```bash
#!/bin/sh
set -eu
ollama serve &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' INT TERM
until ollama list >/dev/null 2>&1; do sleep 2; done
ollama pull nomic-embed-text
ollama pull llama3.2
wait "$ollama_pid"
```

### Step 1 — Build and Start

```bash
docker compose up --build
```

On the **first run** this will:
1. Compile the C++ server inside Docker
2. Start Ollama and pull `nomic-embed-text` (~274 MB) and `llama3.2` (~2 GB)
3. Start the VectorDB server once Ollama is ready

> This takes **3–5 minutes** on first run due to model downloads. Subsequent runs are instant since models are cached in the `ollama_data` volume.

### Step 2 — Open the App

Once you see this in the logs:

```
vectordb-1  | === VectorDB Engine ===
vectordb-1  | http://localhost:8080
vectordb-1  | Ollama: ONLINE
```

Open your browser and go to:

```
http://localhost:8080
```

### Step 3 — Stop

```bash
docker compose down
```

Models stay cached in the volume — next `docker compose up` skips the downloads.

To remove everything including downloaded models:

```bash
docker compose down -v
```

### Troubleshooting Docker

| Problem | Fix |
|---|---|
| `dependency failed to start: ollama is unhealthy` | Ollama took too long to start. Run `docker compose up` again — models are already cached so it will be fast |
| `Ollama: OFFLINE` in the UI | Wait 30 seconds and refresh — vectordb may have started before Ollama was fully ready |
| Port 8080 already in use | Stop whatever is using it: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| LLM answers are slow | Normal on CPU — llama3.2 takes 10–30s. Switch to the 1B model (see below) |

### Use a Smaller/Faster LLM (Optional)

Edit `ollama-entrypoint.sh` and replace `llama3.2` with `llama3.2:1b`:

```bash
ollama pull llama3.2:1b
```

Then edit `main.cpp` and change the gen model:

```cpp
std::string genModel = "llama3.2:1b";
```

Rebuild with `docker compose up --build`.

---

## Running Manually (Windows)

If you prefer to run without Docker, follow the manual setup below.

### Prerequisites

You need **3 things** installed on your Windows laptop:

1. **MSYS2** (gives you g++ compiler)
2. **Git**
3. **Ollama** (runs the local AI models)

### Step 1 — Install MSYS2 (C++ Compiler)

1. Go to **https://www.msys2.org** and download the installer
2. Run the installer, keep default path (`C:\msys64`)
3. After install, open **MSYS2 UCRT64** from Start Menu (the orange icon)
4. Run these commands inside the MSYS2 terminal:

```bash
pacman -Syu
```
*(Close and reopen the terminal if it asks you to)*

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```

5. Add g++ to your Windows PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click **Advanced** → **Environment Variables**
   - Under **System variables**, find **Path**, click **Edit**
   - Click **New** and add: `C:\msys64\ucrt64\bin`
   - Click OK on all windows
   - **Open a new PowerShell** and verify:
   ```
   g++ --version
   ```
   You should see something like `g++ (GCC) 15.x.x`

### Step 2 — Install Git

1. Go to **https://git-scm.com/download/win** and download Git for Windows
2. Run the installer with default settings
3. Verify in PowerShell:
```
git --version
```

### Step 3 — Install Ollama

1. Go to **https://ollama.com** and click **Download for Windows**
2. Run the installer
3. Ollama starts automatically in the system tray
4. Open **PowerShell** and pull the two required models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

5. Verify:
```powershell
ollama list
```

> **Minimum specs:** 8GB RAM recommended. The models will use ~3GB total.

### Step 4 — Clone the Repository

```powershell
git clone https://github.com/dakshjain19/VectorForge-RAG.git
cd VectorForge-RAG
```

### Step 5 — Compile

```powershell
g++ -std=c++17 -O2 -D_WIN32_WINNT=0x0A00 main.cpp -o db -lws2_32
```

### Step 6 — Run

**Terminal 1** — Start Ollama (if not already running):
```powershell
ollama serve
```

**Terminal 2** — Start the VectorDB server:
```powershell
./db
```

Open your browser at `http://localhost:8080`. The service binds to loopback by default. Do not expose this unauthenticated development API to the public internet.

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to benchmark all 3 algorithms

### Tab 2: Documents (Real Embeddings)

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks

### Tab 3: Ask AI (RAG Pipeline)

1. Insert some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based on your documents
```

---

## REST API Reference

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

---

## Project Structure

```
VectorForge-RAG/
├── .github/
│   └── workflows/
│       ├── build.yml
│       └── integration-evaluation.yml
├── benchmarks/
│   └── ann_benchmark.cpp
├── evaluation/
│   ├── README.md
│   ├── rag_eval.json
│   └── rag_eval.py
├── tests/
│   └── api-smoke.sh
├── .dockerignore
├── .gitignore
├── ATTRIBUTION.md
├── Dockerfile
├── docker-compose.yml
├── httplib.h
├── index.html
├── LICENSE
├── main.cpp
├── ollama-entrypoint.sh
└── README.md
```

### Architecture (main.cpp)

```
BruteForce          O(N·d)      Exact, baseline
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer small-world graph

VectorDB            Unified interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client → /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (ef_construction=200) and connect to the M nearest neighbors bidirectionally.

**Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Data and security

Inserted demo vectors and document chunks are persisted in the local `data/` directory across restarts. This directory is ignored by Git. The server binds to `127.0.0.1` by default; Docker Compose publishes port 8080 only on the host loopback interface. Do not expose this unauthenticated development API publicly without adding authentication and transport security.

## License

This project is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the project, subject to the terms of the license.

See the [LICENSE](LICENSE) file for the full license text.

The `httplib.h` dependency also retains its own third-party MIT license and copyright notice. See [ATTRIBUTION.md](ATTRIBUTION.md) for project authorship and third-party notices.


## Evaluation and Scalability

The repository includes reproducible evaluation tools rather than relying only on a successful compile.

### RAG quality evaluation

The golden set in `evaluation/rag_eval.json` contains **five reference documents and five questions** with expected source documents. This is a small smoke/evaluation set, not a statistically representative RAG benchmark. With VectorForge and Ollama running, execute:

```bash
python3 evaluation/rag_eval.py --base-url http://127.0.0.1:8080
```

The evaluator inserts the reference documents, measures retrieval **Hit Rate@K**, **MRR@K**, and **Context Precision@K**, calls the full `/doc/ask` generation route, reports answer-keyword coverage and mean generation time, and deletes the inserted test chunks afterwards. Use `--skip-generation` to evaluate retrieval only. The answer-keyword score is diagnostic rather than a strict pass/fail gate because generated wording can vary between model versions. The retrieval gate requires at least 0.80 Hit Rate@K on this small set; this is not a substitute for a larger human-reviewed benchmark.

### ANN scale and recall benchmark

Compile and run the standalone benchmark:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread benchmarks/ann_benchmark.cpp -o ann-benchmark
./ann-benchmark
```

It builds deterministic **synthetic 16-dimensional datasets** at 1K, 5K, and 10K vectors, runs 50 queries per size, checks KD-Tree cosine results against exact brute force, and reports HNSW Recall@10 plus average query latency and build time. These are local synthetic measurements, not claims about production throughput; repeat on target hardware and representative embedding distributions before drawing performance conclusions.

### Recorded evaluation results

The following results were produced by GitHub Actions run [#2 — VectorForge integration and evaluation](https://github.com/dakshjain19/VectorForge-RAG/actions/runs/42799645823) on **November 10, 2026**. Both the ANN benchmark and full Ollama-backed RAG evaluation jobs completed successfully.

**RAG evaluation** — five documents, five questions, K=3:

| Metric | Result |
|---|---:|
| Hit Rate@3 | 100.00% |
| MRR@3 | 0.90 |
| Context Precision@3 | 33.33% |
| Answer keyword coverage | 73.33% |
| Mean generation time | 18.23 s |

The 100% Hit Rate means each question retrieved its expected source within the top three results. Context Precision is reported separately because relevant retrieval is not the same as every returned context chunk being relevant. With only five questions, these values should be interpreted as a small-set evaluation, not a general quality guarantee.

**ANN benchmark** — synthetic 16D vectors, 50 queries per dataset:

| Dataset size | Build time (ms) | Brute force (µs/query) | KD-Tree (µs/query) | HNSW (µs/query) | HNSW Recall@10 |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 184.87 | 82.99 | 61.02 | 98.18 | 100% |
| 5,000 | 1,913.98 | 455.50 | 310.93 | 156.24 | 100% |
| 10,000 | 4,912.37 | 958.99 | 648.53 | 179.24 | 100% |

At 10K vectors, HNSW's measured average query latency was approximately **5.35× lower** than brute force (179.24 µs vs 958.99 µs). At 1K, HNSW was slower than brute force, so the results do not imply that the approximate index wins at every dataset size. These timings are specific to the GitHub Actions runner and this synthetic workload.

### Cloud integration workflow

GitHub Actions workflow `.github/workflows/integration-evaluation.yml` runs the ANN benchmark and starts the Docker Compose stack for a full Ollama-backed RAG evaluation. It is manually runnable from the Actions tab and also runs when relevant evaluation/runtime files change. The Ollama model downloads can take several minutes and consume substantial runner time.
