#!/usr/bin/env python3
"""
Hierarchical (Paper -> Chunk) Unsupervised Analysis for a RAG Corpus.


python wow.py   --n-paper-clusters 7   --n-chunk-clusters 5   --min-chunks-per-paper 1   --soft-k 4   --anomaly


Input:
- chunks.jsonl (your existing chunk objects from RAG pipeline)
  Each line must be JSON with at least:
    chunk_id, paper_id, title, text, start_word, end_word

Outputs (stdout):
1) Paper-level clusters:
   - Cluster size (papers)
   - Top representative papers (closest to centroid)
   - TF-IDF keywords for cluster (from all chunks in cluster)

2) Soft membership:
   - For each paper: top-k closest clusters + softmax weights

3) Chunk-level clusters inside each paper cluster:
   - Subcluster size (chunks)
   - Representative chunks
   - Subcluster keywords

4) Optional anomaly detection (IsolationForest):
   - Top anomalous chunks per paper cluster

Notes:
- Uses sentence-transformers embeddings; normalized vectors -> cosine similarity via dot product.
- Uses KMeans for clustering (fast, stable).
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest


# =========================
# Data structures
# =========================
import re

BOILERPLATE_PATTERNS = [
    r"https?://\S+",
    r"\bdoi:\s*\S+",
    r"\bdoi\.org/\S+",
    r"\bdownloaded from\b.*",
    r"\borcid\b.*",
    r"\bpubmed\b.*",
    r"\bcrossref\b.*",
    r"\bfigure\s*\d+\b",
    r"\bfig\.\s*\d+\b",
]

def clean_text(t: str) -> str:
    t = t.replace("\u00ad", "")  # soft hyphen
    t = re.sub(r"\s+", " ", t)
    low = t.lower()
    for pat in BOILERPLATE_PATTERNS:
        low = re.sub(pat, " ", low, flags=re.IGNORECASE)
    low = re.sub(r"\b(accepted|received|revised)\b.*", " ", low, flags=re.IGNORECASE)
    low = re.sub(r"\bpage\s*\d+\b", " ", low, flags=re.IGNORECASE)
    low = re.sub(r"\s+", " ", low).strip()
    return low


from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    start_word: int = 0
    end_word: int = 0

    # Newer fields (safe defaults)
    chunk_type: str = "text"   # "text" or "table"
    page_start: int = -1
    page_end: int = -1



def load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


# =========================
# Utilities
# =========================

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    # stable softmax
    z = x / max(temp, 1e-9)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def safe_k(n: int, k: int) -> int:
    return max(1, min(k, n))


def print_hr(char="=", width=80):
    print(char * width)


def top_keywords(texts: List[str], top_n: int = 12) -> List[str]:
    """
    Return top_n TF-IDF keywords for a set of texts.
    """
    if not texts:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # happens when too few docs or empty docs
        return []
    # mean tf-idf score across docs
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    if scores.size == 0:
        return []
    idx = np.argsort(scores)[::-1][:top_n]
    return terms[idx].tolist()


def snippet(text: str, max_chars: int = 280) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + " ..."


# =========================
# Main hierarchical pipeline
# =========================

def build_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32
) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")
    return emb


def compute_paper_embeddings(
    chunks: List[Chunk],
    chunk_emb: np.ndarray,
    min_chunks_per_paper: int = 1
) -> Tuple[List[str], np.ndarray, Dict[str, List[int]]]:
    """
    Returns:
      paper_ids: list of paper_ids in fixed order
      paper_emb: (n_papers, dim) normalized
      paper_to_chunk_idxs: mapping paper_id -> list of chunk indices
    """
    paper_to_chunk_idxs: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        paper_to_chunk_idxs.setdefault(c.paper_id, []).append(i)

    # Filter papers with too few chunks
    paper_ids = []
    paper_vecs = []

    for pid, idxs in paper_to_chunk_idxs.items():
        if len(idxs) < min_chunks_per_paper:
            continue
        paper_ids.append(pid)
        v = chunk_emb[idxs].mean(axis=0)
        paper_vecs.append(v)

    paper_emb = np.vstack(paper_vecs).astype("float32")
    paper_emb = normalize_rows(paper_emb)

    # Also filter mapping to only included papers
    paper_to_chunk_idxs = {pid: paper_to_chunk_idxs[pid] for pid in paper_ids}

    return paper_ids, paper_emb, paper_to_chunk_idxs


def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      labels: (n,)
      centroids: (k, dim) normalized
    """
    n = X.shape[0]
    k = safe_k(n, n_clusters)
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_.astype("float32")
    centroids = normalize_rows(centroids)
    return labels, centroids


def paper_soft_membership(
    paper_emb: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 3,
    temp: float = 0.07
) -> List[List[Tuple[int, float]]]:
    """
    For each paper embedding, compute similarity to all centroids (cosine via dot),
    pick top_k, return softmax weights over those.
    """
    sims = paper_emb @ centroids.T  # (n_papers, k)
    memberships = []
    for i in range(sims.shape[0]):
        row = sims[i]
        k = safe_k(row.size, top_k)
        top_idx = np.argsort(row)[::-1][:k]
        weights = softmax(row[top_idx], temp=temp)
        memberships.append([(int(ci), float(w)) for ci, w in zip(top_idx, weights)])
    return memberships


def representative_indices(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    top_n: int = 5
) -> Dict[int, List[int]]:
    """
    For each cluster, find top_n points closest to centroid (highest cosine sim).
    """
    reps: Dict[int, List[int]] = {}
    for c in range(centroids.shape[0]):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            reps[c] = []
            continue
        sims = X[idxs] @ centroids[c]
        order = np.argsort(sims)[::-1][:top_n]
        reps[c] = idxs[order].tolist()
    return reps


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.02,
    seed: int = 42
) -> np.ndarray:
    """
    Returns anomaly scores (lower -> more anomalous).
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed
    )
    iso.fit(X)
    # decision_function: higher = more normal; lower = more anomalous
    scores = iso.decision_function(X)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl", help="Path to chunks.jsonl")
    ap.add_argument("--embed-model", default="all-mpnet-base-v2", help="SentenceTransformer model")
    ap.add_argument("--batch-size", type=int, default=32)

    ap.add_argument("--n-paper-clusters", type=int, default=12)
    ap.add_argument("--n-chunk-clusters", type=int, default=6)

    ap.add_argument("--top-papers", type=int, default=5, help="Representative papers per paper-cluster")
    ap.add_argument("--top-chunks", type=int, default=5, help="Representative chunks per chunk-subcluster")

    ap.add_argument("--soft-k", type=int, default=3, help="Top-k clusters for soft membership per paper")
    ap.add_argument("--soft-temp", type=float, default=0.07, help="Softmax temperature for membership weights")

    ap.add_argument("--min-chunks-per-paper", type=int, default=2)

    ap.add_argument("--keywords", type=int, default=12, help="Top TF-IDF keywords per cluster")
    ap.add_argument("--max-chars", type=int, default=300, help="Max snippet chars")

    ap.add_argument("--anomaly", action="store_true", help="Run IsolationForest per paper-cluster")
    ap.add_argument("--anomaly-contamination", type=float, default=0.02)
    ap.add_argument("--top-anomalies", type=int, default=8)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"Missing chunks file: {args.chunks}")

    print("[load] Loading chunks...")
    chunks = load_chunks_jsonl(args.chunks)
    print(f"[load] {len(chunks)} chunks loaded.")

    print("[embed] Loading embedding model:", args.embed_model)
    emb_model = SentenceTransformer(args.embed_model)

    print("[embed] Embedding chunks (normalized)...")
    chunk_texts = [clean_text(c.text) for c in chunks]
    chunk_emb = build_embeddings(emb_model, chunk_texts, batch_size=args.batch_size)

    print("[paper] Building paper embeddings...")
    paper_ids, paper_emb, paper_to_chunk_idxs = compute_paper_embeddings(
        chunks, chunk_emb, min_chunks_per_paper=args.min_chunks_per_paper
    )
    print(f"[paper] Papers included: {len(paper_ids)} (min_chunks_per_paper={args.min_chunks_per_paper})")

    # Map paper_id -> title (first chunk title)
    paper_title: Dict[str, str] = {}
    for c in chunks:
        if c.paper_id not in paper_title:
            paper_title[c.paper_id] = c.title

    # =========================
    # 1) Paper-level clustering
    # =========================
    print("\n[paper] Clustering papers...")
    paper_labels, paper_centroids = kmeans_cluster(paper_emb, args.n_paper_clusters, seed=args.seed)
    paper_reps = representative_indices(paper_emb, paper_labels, paper_centroids, top_n=args.top_papers)

    # Soft membership (mixture) for each paper
    memberships = paper_soft_membership(
        paper_emb, paper_centroids, top_k=args.soft_k, temp=args.soft_temp
    )

    # Invert cluster -> paper indices
    cluster_to_papers: Dict[int, List[int]] = {}
    for i, lab in enumerate(paper_labels):
        cluster_to_papers.setdefault(int(lab), []).append(i)

    # Print paper clusters
    print_hr("=")
    print("PAPER-LEVEL CLUSTERS")
    print_hr("=")

    for cl in sorted(cluster_to_papers.keys()):
        idxs = cluster_to_papers[cl]
        print(f"\nCLUSTER {cl}  (papers={len(idxs)})")
        print_hr("-")

        # Keywords for cluster from ALL chunk texts in these papers
        cluster_chunk_texts = []
        for pi in idxs:
            pid = paper_ids[pi]
            for ci in paper_to_chunk_idxs[pid]:
                cluster_chunk_texts.append(chunks[ci].text)
        kws = top_keywords(cluster_chunk_texts, top_n=args.keywords)
        if kws:
            print("keywords:", ", ".join(kws))

        # Representative papers
        rep = paper_reps.get(cl, [])
        if rep:
            print("\nRepresentative papers:")
            for pi in rep:
                pid = paper_ids[pi]
                title = paper_title.get(pid, pid)
                print(f"  - {title} ({pid})")

        # Show a few papers with mixture membership (closest ones)
        print("\nExample soft memberships (top few papers in this cluster):")
        # take first up to 5 papers in cluster
        for pi in idxs[:5]:
            pid = paper_ids[pi]
            title = paper_title.get(pid, pid)
            mix = memberships[pi]
            mix_str = " ".join([f"C{c}:{w:.2f}" for c, w in mix])
            print(f"  - {title} ({pid})  ->  {mix_str}")

    # =========================
    # 2) Chunk-level clustering inside each paper cluster
    # =========================
    print("\n")
    print_hr("=")
    print("CHUNK-LEVEL CLUSTERS WITHIN EACH PAPER CLUSTER")
    print_hr("=")

    for cl in sorted(cluster_to_papers.keys()):
        paper_idxs = cluster_to_papers[cl]
        # Collect chunk indices belonging to these papers
        chunk_idxs = []
        for pi in paper_idxs:
            pid = paper_ids[pi]
            chunk_idxs.extend(paper_to_chunk_idxs[pid])

        if len(chunk_idxs) < max(10, args.n_chunk_clusters * 3):
            print(f"\n[cluster {cl}] Skipping chunk-subclustering (too few chunks: {len(chunk_idxs)})")
            continue

        X = chunk_emb[chunk_idxs]  # (n_chunks_in_cluster, dim)
        sub_labels, sub_centroids = kmeans_cluster(X, args.n_chunk_clusters, seed=args.seed)
        sub_reps = representative_indices(X, sub_labels, sub_centroids, top_n=args.top_chunks)

        # Group subcluster -> local indices (0..len(chunk_idxs)-1)
        sub_to_local: Dict[int, List[int]] = {}
        for li, lab in enumerate(sub_labels):
            sub_to_local.setdefault(int(lab), []).append(li)

        print(f"\nPAPER CLUSTER {cl}  (chunks={len(chunk_idxs)})")
        print_hr("-")

        # Optional anomaly detection within this paper cluster
        anomaly_scores = None
        if args.anomaly and len(chunk_idxs) >= 50:
            print(f"[anomaly] Running IsolationForest (contamination={args.anomaly_contamination})...")
            anomaly_scores = run_isolation_forest(X, contamination=args.anomaly_contamination, seed=args.seed)

        # Print subclusters
        for sub in sorted(sub_to_local.keys()):
            local_list = sub_to_local[sub]
            print(f"\n  subcluster {sub}  (chunks={len(local_list)})")

            # Keywords for subcluster
            sub_texts = [chunks[chunk_idxs[li]].text for li in local_list]
            kws = top_keywords(sub_texts, top_n=args.keywords)
            if kws:
                print("    keywords:", ", ".join(kws))

            # Representative chunks
            reps_local = sub_reps.get(sub, [])
            if reps_local:
                print("    representative chunks:")
                for li in reps_local:
                    gi = chunk_idxs[li]
                    c = chunks[gi]
                    print(f"      - {c.title} ({c.paper_id}) [{c.chunk_id}]")
                    print(f"        {snippet(c.text, max_chars=args.max_chars)}")

        # Print anomalies if requested
        if anomaly_scores is not None:
            order = np.argsort(anomaly_scores)[:args.top_anomalies]
            print("\n  [anomaly] Top anomalous chunks (potentially novel/contradictory/off-topic):")
            for li in order:
                gi = chunk_idxs[li]
                c = chunks[gi]
                print(f"    - score={anomaly_scores[li]:.4f}  {c.title} ({c.paper_id}) [{c.chunk_id}]")
                print(f"      {snippet(c.text, max_chars=args.max_chars)}")

    # =========================
    # 3) Print full soft membership table (optional-ish)
    # =========================
    print("\n")
    print_hr("=")
    print("SOFT MEMBERSHIP (ALL PAPERS)")
    print_hr("=")
    for i, pid in enumerate(paper_ids):
        title = paper_title.get(pid, pid)
        mix = memberships[i]
        mix_str = " ".join([f"C{c}:{w:.2f}" for c, w in mix])
        print(f"- {title} ({pid})  ->  {mix_str}")


if __name__ == "__main__":
    main()



