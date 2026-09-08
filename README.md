# 🚀 Qdrant Enterprise Vector Search Project
### *Production-Grade 3072-Dimensional Vector Database with OpenAI `text-embedding-3-large`*

---

## 📌 1. Project Overview

This project provides an enterprise vector search architecture built with **Qdrant** and OpenAI's flagship **`text-embedding-3-large`** model operating at **3,072 dimensions**.

It replaces lightweight in-memory vector stores (like ChromaDB) with a high-throughput, Rust-powered vector database engineered for enterprise RAG, high-concurrency semantic search, and metadata pre-filtering.

---

## ⚖️ 2. Enterprise Comparison: Qdrant vs. ChromaDB

| Feature | **Qdrant** (Enterprise Production Standard) 🏆 | **ChromaDB** (Prototyping / Local Dev) |
| :--- | :--- | :--- |
| **Engine Core** | Pure **Rust** engine with SIMD vector hardware acceleration | Python runtime wrapping SQLite / DuckDB / Clickhouse |
| **Embedding Dimension Capacity** | Native **3072-dim** support with Scalar (SQ) and Product Quantization (PQ) | In-memory indexing with higher RAM pressure on large dimensions |
| **Filtering Mechanism** | **Payload Pre-Filtering** inside HNSW graph nodes (no empty search results) | Post-filtering or external database joins |
| **Storage Architecture** | **Memory-Mapped Storage (Mmap on NVMe)**: Keeps vectors on disk, low RAM footprint | In-memory vector indexes |
| **Clustering & High Availability** | Distributed clustering with **Raft consensus**, auto-sharding, and replication | Single-node focus |
| **Production Features** | Snapshot backups, RBAC access tokens, zero-downtime collection resizing | Simpler setup for rapid notebook experimentation |

---

## 📐 3. Why 3,072 Dimensions? (`text-embedding-3-large`)

* **Legacy / Small Embeddings (768 or 1536 dims):** Good for generic matching, but loses subtle technical, legal, and multi-domain nuances.
* **Flagship 3072-Dimensional Embeddings:** Captures rich semantic density and complex cross-lingual context with state-of-the-art retrieval accuracy.
* **Qdrant's Role:** Storing 3072 numbers per vector requires 4x more memory than 768 dims. Qdrant solves this with **Mmap on-disk storage** and **quantization**, keeping server costs low while maximizing search accuracy.

---

## 🚀 4. Quick Start Guide

### Step 1: Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Interactive CLI Demo
```bash
python demo.py
```

### Step 4: Run the FastAPI Web Service & Swagger UI
```bash
python api/main.py
```
Open **http://127.0.0.1:8000/docs** in your browser to test endpoints interactively!

---

## ⚙️ 5. Configuration (`.env`)

Create a `.env` file from `.env.example`:

```ini
# OpenAI API Key (Leave empty to use built-in 3072-dim mock vector generator for offline testing)
OPENAI_API_KEY=sk-your-openai-key-here

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

# Qdrant Database Configuration
# Mode A: Local Persistent Storage (Default)
QDRANT_STORAGE_PATH=./qdrant_storage
QDRANT_COLLECTION_NAME=enterprise_knowledge_base

# Mode B: Qdrant Cloud Cluster (If using Qdrant Cloud Free/Pro Cluster)
# QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
# QDRANT_API_KEY=your-qdrant-api-key-here
```

---

## 🧪 6. Running Automated Tests
```bash
pytest tests/test_qdrant.py -v
```
