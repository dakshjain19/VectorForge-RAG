# VectorForge-RAG — C++ Vector Database & Local RAG

> **Created and maintained by Daksh Jain** ([@dakshjain19](https://github.com/dakshjain19)).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A fully working **Vector Database** built from scratch in C++ with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

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

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Running with Docker (Recommended)

Docker is the easiest way to run this project — no need to install g++, MSYS2, or manually pull Ollama models.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Project Structure

Make sure your project folder looks like this:

```
VectorForge-RAG/
├── main.cpp
├── httplib.h
├── index.html
├── Dockerfile
├── docker-compose.yml
└── ollama-entrypoint.sh
```

### Dockerfile

```dockerfile
# Stage 1: Build
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN g++ -std=c++17 -O2 main.cpp -o db -lpthread

# Stage 2: Run
FROM ubuntu:22.04 AS runner

WORKDIR /app
COPY --from=builder /app/db .
COPY --from=builder /app/index.html .

EXPOSE 8080
CMD ["./db"]
```

### docker-compose.yml

```yaml
services:
  vectordb:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OLLAMA_HOST=ollama
    depends_on:
      ollama:
        condition: service_healthy

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      - ./ollama-entrypoint.sh:/entrypoint.sh
    entrypoint: ["/bin/sh", "/entrypoint.sh"]
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 15s

volumes:
  ollama_data:
```

### ollama-entrypoint.sh

```bash
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

Open your browser at `http://localhost:8080`.

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
VectorDB/
├── main.cpp                ← C++ backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── httplib.h               ← Single-header HTTP server library (cpp-httplib)
├── index.html              ← Frontend (PCA scatter plot, chat UI, benchmark)
├── Dockerfile              ← Multi-stage build (Ubuntu builder → runner)
├── docker-compose.yml      ← Wires vectordb + ollama together
├── ollama-entrypoint.sh    ← Auto-pulls models on first run
└── README.md               ← This file
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

## License

This project is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the project, subject to the terms of the license.

See the [LICENSE](LICENSE) file for the full license text.

The `httplib.h` dependency also retains its own third-party MIT license and copyright notice. See [ATTRIBUTION.md](ATTRIBUTION.md) for project authorship and third-party notices.
