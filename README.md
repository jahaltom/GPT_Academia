# CMEM RAG Prototype (Qdrant + Mistral 7B) — HPC Setup Guide

This README documents **every step** required to run the CMEM Retrieval‑Augmented Generation (RAG) prototype on an HPC cluster **without Docker**, using **Qdrant** as the vector database and a local **Mistral‑7B** model.


---

## Overview

**Architecture**

- PDF ingestion → clean text + tables
- Sentence‑aware chunking with overlap
- Embeddings: PubMedBERT (MS‑MARCO tuned)
- Vector store: **Qdrant (local server, no Docker)**
- Retrieval: dense + BM25 + reranking + neighbor expansion
- Generation: **Mistral‑7B‑Instruct** (local)

**Key constraint**

> Qdrant **must run on the same compute node** as `rag_hpc.py`.

---

## 0) One‑time directory setup

```bash
mkdir -p /scr1/users/haltomj/ChatGPT_Acadamia/{bin,logs,qdrant_data,pdfs,parsed,index}
```

---

## 1) One‑time: Download & install Qdrant (no Docker)

Download the Linux x86_64 binary directly from GitHub.

```bash
cd /scr1/users/haltomj/ChatGPT_Acadamia/bin
VER="1.15.3"

# download musl/static build (name can vary slightly by release)
curl -L -o qdrant_musl.tar.gz \
  "https://github.com/qdrant/qdrant/releases/download/v${VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"

tar -xzf qdrant_musl.tar.gz
rm -f qdrant_musl.tar.gz

# verify it runs
./qdrant --version
```

> If this fails due to glibc incompatibilities, download the **musl / static** build from the same release page and repeat the steps.

---

## 2) One‑time: Create Qdrant config

Create:

```
/scr1/users/haltomj/ChatGPT_Acadamia/qdrant.yaml
```

```yaml
service:
  host: 127.0.0.1
  http_port: 6333

storage:
  storage_path: /scr1/users/haltomj/ChatGPT_Acadamia/qdrant_data
```

---






---

## 4) Interactive workflow (recommended for live Q&A)

### 4.1 Start tmux (protect against SSH disconnects)

```bash
tmux new -s rag
```

Detach: `Ctrl-b d`  
Reattach: `tmux attach -t rag`

---

### 4.2 Allocate a compute node

```bash
srun --pty -N 1 -n 1 -c 12 --mem=32G --gres=gpu:1 -p gpuq -t 08:00:00 bash -l
```

---

### 4.3 Activate environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag
cd /scr1/users/haltomj/ChatGPT_Acadamia
```

---

### 4.4 Start Qdrant (background)

```bash
/scr1/users/haltomj/ChatGPT_Acadamia/bin/qdrant \
  --config-path /scr1/users/haltomj/ChatGPT_Acadamia/qdrant.yaml \
  > /scr1/users/haltomj/ChatGPT_Acadamia/logs/qdrant.log 2>&1 &
```

Verify (critical step):

```bash
curl -s http://127.0.0.1:6333/collections
```

Expected output:

```json
{"status":"ok","result":{"collections":[]}}
```

---

### 4.5 Run the RAG pipeline

**First run (build index + upload vectors):**

```bash
python /scr1/users/haltomj/ChatGPT_Acadamia/rag_hpc.py \
  --qdrant_host 127.0.0.1 \
  --qdrant_port 6333 \
  --qdrant_collection cmem_chunks 
```

You will then see:

```
Question:
```

Ask questions interactively.

---

### 4.6 Stop Qdrant (optional)

```bash
ss -ltnp | grep 6333
kill <PID>
```

---

## 5) Batch workflow (index build only)

This is useful for rebuilding embeddings **without interaction**.

Create:

```
/scr1/users/haltomj/ChatGPT_Acadamia/run_build_qdrant.sbatch
```

```bash
#!/bin/bash
#SBATCH --job-name=rag_qdrant_build
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scr1/users/haltomj/ChatGPT_Acadamia/logs/rag_build.%j.out
#SBATCH --error=/scr1/users/haltomj/ChatGPT_Acadamia/logs/rag_build.%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

cd /scr1/users/haltomj/ChatGPT_Acadamia

# Start Qdrant
/scr1/users/haltomj/ChatGPT_Acadamia/bin/qdrant \
  --config-path /scr1/users/haltomj/ChatGPT_Acadamia/qdrant.yaml \
  > /scr1/users/haltomj/ChatGPT_Acadamia/logs/qdrant.$SLURM_JOB_ID.log 2>&1 &

QPID=$!

# Wait until Qdrant responds
for i in {1..60}; do
  if curl -s http://127.0.0.1:6333/collections >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Build index
python /scr1/users/haltomj/ChatGPT_Acadamia/rag_hpc.py \
  --backend qdrant \
  --qdrant_host 127.0.0.1 \
  --qdrant_port 6333 \
  --qdrant_collection cmem_chunks \
  --rebuild

# Stop Qdrant
kill $QPID || true
```

Submit:

```bash
sbatch /scr1/users/haltomj/ChatGPT_Acadamia/run_build_qdrant.sbatch
```

---

## 6) Critical rules (document these)

1. **Qdrant and Python must run on the same node**  
   Do not start Qdrant on a login node and Python on a compute node.

2. **Interactive questions require `srun --pty`**  
   Batch jobs cannot accept input at the `Question:` prompt.

3. **Large indexes require batched upserts**  
   The code already batches Qdrant uploads to avoid HTTP timeouts.

---

## 7) Troubleshooting

- **Qdrant not responding** → check `logs/qdrant.log`
- **Connection refused** → Qdrant not running or wrong node
- **Timeout during upsert** → reduce batch size or increase client timeout
- **tmux/vim slow** → ensure `TERM=screen-256color` and disable syntax for large files

---

