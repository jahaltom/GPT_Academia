#!/usr/bin/env python3
"""
End-to-end RAG + Mistral 7B over PDFs (HPC-friendly, local-only).

Built directly off your script, but adds the fixes you asked for:

1) Better chunking to avoid "answer is in the previous chunk":
   - Sentence-aware chunking (with sentence overlap)
   - Optional smaller word-window fallback

2) Hybrid retrieval:
   - Dense (SentenceTransformers / PubMedBERT MS-MARCO) + FAISS
   - Lexical BM25 (rank-bm25)
   - Candidate merge + optional CrossEncoder rerank (ms-marco MiniLM)

3) Query-aware weighting:
   - If question is RNA-focused, boost RNA-like chunks and downrank protein-only chunks.

4) Neighbor-chunk expansion:
   - If we retrieve chunk_N, we also auto-include chunk_(N-1) and chunk_(N+1) from same paper
     to catch exact sentences that fell on boundary.

5) Output validation + retry:
   - Enforces 8–20 FACTS bullets, each bullet starts with exactly ONE valid [Paper_*.pdf] tag.

Notes:
- Still uses pdfplumber parsing + optional table extraction
- Still writes parsed/<paper_id>.json
- Still writes chunks.jsonl and index files
- Still uses ONLY [Paper_<filename.pdf>] tags

Optional deps (recommended but optional):
  pip install rank-bm25
  pip install sentence-transformers
  pip install faiss-cpu  (or faiss-gpu)
  pip install transformers torch
  pip install pdfplumber
  pip install cross-encoder  (actually comes via sentence-transformers), but model download needed

If CrossEncoder isn't available / can't download, script will run without reranking.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import math
import numpy as np

import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
# Optional BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# Optional CrossEncoder reranker
try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except Exception:
    HAS_CE = False

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# =========================
# Config / paths
# =========================

DATA_DIR = "/scr1/users/haltomj/ChatGPT_Acadamia"

PDF_DIR = os.path.join(DATA_DIR, "pdfs")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
INDEX_DIR = os.path.join(DATA_DIR, "index")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.json")               # tokenized corpus + mapping
EMB_CACHE_PATH = os.path.join(INDEX_DIR, "embeddings.npy")     # cache dense embeddings

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Embedding model (your choice)
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Optional CrossEncoder reranker (small + strong)
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Local LLM
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


# =========================
# 0. Load Mistral 7B
# =========================

print("[llm] Loading Mistral 7B model...")

tokenizer_llm = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
model_llm = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

llm_pipeline = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer_llm,
    max_new_tokens=1024,
    do_sample=False,
)

print("[llm] Mistral 7B loaded.")


# =========================
# Data classes
# =========================

@dataclass
class Paper:
    paper_id: str
    title: str           # filename.pdf
    text: str


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    start_word: int
    end_word: int
    chunk_type: str      # "text" or "table"
    page_start: int
    page_end: int

    # NEW: backward compatible defaults
    chunk_index: int = -1


# =========================
# Reference / boilerplate chunk filter
# =========================

def looks_like_references(text: str) -> bool:
    """
    Conservative heuristic to detect reference sections / bibliographies / citation lists.
    """
    if not text:
        return True

    t = text.strip().lower()

    if re.search(r"(^|\n)\s*(references|bibliography|literature cited)\s*($|\n)", t):
        return True

    doi_hits = len(re.findall(r"\bdoi\b|10\.\d{4,9}/[-._;()/:a-z0-9]+", t))
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", t))
    etal_hits = t.count("et al")
    url_hits = t.count("http://") + t.count("https://")
    journalish = len(re.findall(r"\b(journal|nature|science|cell|proc\.|proceedings|biorxiv|medrxiv)\b", t))

    if (year_hits >= 12 and (doi_hits >= 2 or etal_hits >= 5)) or (doi_hits >= 4):
        return True
    if url_hits >= 4 and year_hits >= 8:
        return True
    if year_hits >= 15 and journalish >= 4:
        return True

    return False


# =========================
# 1. PDF parsing (pdfplumber)
# =========================

def _table_to_markdown(table: List[List[Any]]) -> str:
    if not table:
        return ""
    rows = []
    for r in table:
        if r is None:
            continue
        rows.append([("" if c is None else str(c)).strip() for c in r])

    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    header = rows[0]
    sep = ["---"] * max_cols
    body = rows[1:] if len(rows) > 1 else []

    def fmt_row(r):
        return "| " + " | ".join(r) + " |"

    md = [fmt_row(header), fmt_row(sep)]
    for r in body:
        md.append(fmt_row(r))
    return "\n".join(md)


def parse_pdf_to_text_and_tables(pdf_path: str, extract_tables: bool = True) -> Dict[str, Any]:
    page_texts: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""

            if looks_like_references(txt):
                continue

            page_texts.append({"page": i, "text": txt})

            if extract_tables:
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []

                for t_idx, t in enumerate(raw_tables):
                    md = _table_to_markdown(t)
                    if md.strip():
                        tables.append({
                            "page": i,
                            "table_index": t_idx,
                            "markdown": md
                        })

    full_text = "\n\n".join([pt["text"] for pt in page_texts])

    return {"full_text": full_text, "page_texts": page_texts, "tables": tables}


def parse_all_pdfs(pdf_dir: str, extract_tables: bool = True) -> List[Paper]:
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    papers: List[Paper] = []
    for fname in sorted(pdf_files):
        pdf_path = os.path.join(pdf_dir, fname)
        print(f"[parse] Parsing {pdf_path} ...")

        parsed = parse_pdf_to_text_and_tables(pdf_path, extract_tables=extract_tables)

        title = os.path.basename(pdf_path)
        paper_id = os.path.splitext(title)[0]

        paper = Paper(paper_id=paper_id, title=title, text=parsed["full_text"])

        parsed_out = os.path.join(PARSED_DIR, f"{paper.paper_id}.json")
        with open(parsed_out, "w", encoding="utf-8") as f:
            json.dump(
                {**asdict(paper), "page_texts": parsed["page_texts"], "tables": parsed["tables"]},
                f,
                ensure_ascii=False,
                indent=2,
            )

        papers.append(paper)

    print(f"[parse] Parsed {len(papers)} papers.")
    return papers


# =========================
# 2. Chunking (sentence-aware + neighbor-safe overlap)
# =========================

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (good enough for papers).
    """
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    # prevent breaking on common abbreviations
    t = re.sub(r"\b(e\.g|i\.e|et al)\.\s", lambda m: m.group(0).replace(". ", "<DOT> "), t, flags=re.I)
    parts = _SENT_SPLIT.split(t)
    parts = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return parts


def chunk_paper_text_sentence_aware(
    paper: Paper,
    max_words: int = 450,
    overlap_sents: int = 4,
) -> List[Chunk]:
    """
    Packs sentences into chunks up to max_words.
    Overlaps by overlap_sents to avoid boundary misses.
    """
    sents = split_into_sentences(paper.text)
    if not sents:
        return []

    chunks: List[Chunk] = []
    i = 0
    chunk_idx = 0

    # Build a mapping from sentence -> word counts
    sent_words = [len(s.split()) for s in sents]

    while i < len(sents):
        w = 0
        j = i
        buf = []
        start_word = None
        # approximate word positions by cumulative counts
        # (we keep start_word/end_word for debugging; exactness isn't critical)
        while j < len(sents) and (w + sent_words[j]) <= max_words:
            if start_word is None:
                start_word = sum(sent_words[:j])
            buf.append(sents[j])
            w += sent_words[j]
            j += 1

        if not buf:
            # single huge sentence; force it
            start_word = sum(sent_words[:i])
            buf = [sents[i]]
            j = i + 1

        text = " ".join(buf).strip()
        end_word = (start_word or 0) + w

        if not looks_like_references(text):
            chunk_id = f"{paper.paper_id}_textchunk_{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    text=text,
                    start_word=int(start_word or 0),
                    end_word=int(end_word),
                    chunk_type="text",
                    page_start=-1,
                    page_end=-1,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

        # overlap: back up overlap_sents from j
        i = max(j - overlap_sents, i + 1)

    return chunks


def load_tables_from_parsed_json(paper_id: str) -> List[Dict[str, Any]]:
    parsed_out = os.path.join(PARSED_DIR, f"{paper_id}.json")
    if not os.path.exists(parsed_out):
        return []
    with open(parsed_out, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tables", []) or []


def chunk_paper_tables(paper: Paper) -> List[Chunk]:
    tables = load_tables_from_parsed_json(paper.paper_id)
    chunks: List[Chunk] = []
    for i, t in enumerate(tables):
        md = (t.get("markdown") or "").strip()
        if not md:
            continue
        chunk_id = f"{paper.paper_id}_table_{i}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper.paper_id,
                title=paper.title,
                text=md,
                start_word=0,
                end_word=0,
                chunk_type="table",
                page_start=int(t.get("page", -1)),
                page_end=int(t.get("page", -1)),
                chunk_index=-1,
            )
        )
    return chunks


def build_all_chunks(
    papers: List[Paper],
    max_words: int = 450,
    overlap_sents: int = 4,
    include_tables: bool = True,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for p in papers:
        all_chunks.extend(chunk_paper_text_sentence_aware(p, max_words=max_words, overlap_sents=overlap_sents))
        if include_tables:
            all_chunks.extend(chunk_paper_tables(p))
    print(f"[chunk] Built {len(all_chunks)} chunks across {len(papers)} papers.")
    return all_chunks


def save_chunks_jsonl(chunks: List[Chunk], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"[chunk] Saved chunks to {out_path}")


def load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Backward-compat: older chunks don't have chunk_index
            if "chunk_index" not in data:
                # If it's a text chunk, try to infer from chunk_id suffix
                # e.g. paper_textchunk_12
                ci = -1
                if data.get("chunk_type") == "text":
                    m = re.search(r"_textchunk_(\d+)$", data.get("chunk_id", ""))
                    if m:
                        ci = int(m.group(1))
                data["chunk_index"] = ci

            chunks.append(Chunk(**data))

    print(f"[chunk] Loaded {len(chunks)} chunks from {path}")
    return chunks



# =========================
# 3. Index: Dense + BM25 + optional CrossEncoder rerank
# =========================



_word_tok = re.compile(r"[A-Za-z0-9_+\-Δ]+")  # keep some science tokens

def bm25_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_tok.findall(text or "")]



def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default




class QdrantRagIndex:
    """
    Qdrant-backed retrieval with payload metadata.
    Keeps the same output schema as your current rag.search().
    """

    def __init__(self, model_name: str, device: str = "cpu",
                 host: str = "localhost", port: int = 6333,
                 collection: str = "cmem_chunks",
                 use_bm25: bool = True, use_cross_encoder: bool = False):

        self.model_name = model_name
        self.device = device
        print(f"[embed] Loading SentenceTransformer: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.client = QdrantClient(host=host, port=port, timeout=120.0)
        self.collection = collection

        self.use_bm25 = bool(use_bm25 and HAS_BM25)
        self.use_cross_encoder = bool(use_cross_encoder and HAS_CE)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME) if self.use_cross_encoder else None

        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

        # optional BM25 (still local, because Qdrant isn’t BM25)
        self.bm25 = None
        self.bm25_tokens: List[List[str]] = []

        # neighbor lookup for text chunks
        self.text_lookup: Dict[Tuple[str, int], int] = {}

    def _build_text_lookup(self):
        self.text_lookup = {}
        for gi, c in enumerate(self.chunks):
            if c.chunk_type == "text" and c.chunk_index >= 0:
                self.text_lookup[(c.paper_id, c.chunk_index)] = gi

    def build_index(self, chunks: List[Chunk], cache_embeddings: bool = True, recreate: bool = True) -> None:
        self.chunks = chunks
        self._build_text_lookup()

        texts = [c.text for c in chunks]
        print(f"[index] Embedding {len(texts)} chunks with {self.model_name} (normalized)...")
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")
        self.embeddings = embeddings

        if cache_embeddings:
            np.save(EMB_CACHE_PATH, embeddings)

        dim = int(embeddings.shape[1])

        if recreate:
            print(f"[qdrant] Recreating collection '{self.collection}' (dim={dim})...")
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

        print("[qdrant] Upserting points in batches...")
        
        BATCH = 256  # 128–1024 is typical; start with 256
        for start in range(0, len(chunks), BATCH):
            end = min(start + BATCH, len(chunks))
            points = []
            for i in range(start, end):
                c = chunks[i]
                payload = {
                    "gid": i,
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "title": c.title,
                    "chunk_type": c.chunk_type,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "chunk_index": c.chunk_index,
                    # If you store text, keep it, but note payload size:
                    "text": c.text,
                }
                points.append(qm.PointStruct(id=i, vector=embeddings[i].tolist(), payload=payload))
        
            # wait=True makes the call synchronous; can set wait=False for faster, but start with True
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        
            if (start // BATCH) % 10 == 0:
                print(f"[qdrant] Upserted {end}/{len(chunks)} points...")


        # local BM25 (optional)
        if self.use_bm25:
            print("[bm25] Building BM25 corpus...")
            self.bm25_tokens = [bm25_tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(self.bm25_tokens)
            with open(BM25_PATH, "w", encoding="utf-8") as f:
                json.dump({"tokens": self.bm25_tokens}, f)
            print("[bm25] BM25 built.")

        print("[qdrant] Index ready.")

    def load(self, chunks: List[Chunk]) -> None:
        """
        Qdrant stores vectors; we still keep chunks in memory for prompt building.
        """
        self.chunks = chunks
        self._build_text_lookup()

        if os.path.exists(EMB_CACHE_PATH):
            self.embeddings = np.load(EMB_CACHE_PATH).astype("float32")

        # BM25 optional
        if self.use_bm25 and os.path.exists(BM25_PATH):
            with open(BM25_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.bm25_tokens = data["tokens"]
            self.bm25 = BM25Okapi(self.bm25_tokens)

        print(f"[qdrant] Loaded chunks ({len(self.chunks)}).")

    def _qdrant_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[qm.Filter]:
        if not filter_dict:
            return None

        must = []
        # supports {"paper_id": "...", "chunk_type": "table"} exact matches
        for k, v in filter_dict.items():
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        return qm.Filter(must=must)

    def _bm25_candidates(self, query: str, fetch_k: int) -> List[int]:
        if not self.use_bm25 or self.bm25 is None:
            return []
        qtok = bm25_tokenize(query)
        scores = self.bm25.get_scores(qtok)
        idxs = np.argsort(scores)[::-1][:fetch_k]
        return [int(i) for i in idxs if scores[i] > 0]

    def _add_neighbors(self, idxs: List[int], neighbor_window: int = 3) -> List[int]:
        out = set(idxs)
        for gi in idxs:
            c = self.chunks[gi]
            if c.chunk_type != "text" or c.chunk_index < 0:
                continue
            for d in range(1, neighbor_window + 1):
                for ni in (c.chunk_index - d, c.chunk_index + d):
                    gni = self.text_lookup.get((c.paper_id, ni))
                    if gni is not None:
                        out.add(gni)
        return list(out)

    def _crossencoder_rerank(self, query: str, idxs: List[int]) -> List[int]:
        if not self.use_cross_encoder or self.cross_encoder is None:
            return idxs
        pairs = [(query, self.chunks[i].text[:2000]) for i in idxs]
        scores = self.cross_encoder.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [idxs[int(i)] for i in order]

    def search(self, query: str, top_k: int = 12,
               dense_fetch_k: int = 200, bm25_fetch_k: int = 200,
               neighbor_window: int = 1, allow_rerank: bool = True,
               meta_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

        # Query embedding
        q_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]

        qfilter = self._qdrant_filter(meta_filter)

        limit = min(dense_fetch_k, len(self.chunks))
        
        # Newer qdrant-client uses query_points(); older uses search()
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection,
                query=q_vec.tolist(),          # vector
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
            hits = resp.points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=q_vec.tolist(),
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
            )


        dense_idxs = []
        dense_scores = {}
        for h in hits:
            gi = int(h.payload.get("gid"))
            dense_idxs.append(gi)
            dense_scores[gi] = float(h.score)

        bm25 = self._bm25_candidates(query, min(bm25_fetch_k, len(self.chunks)))
        merged = list(dict.fromkeys(dense_idxs + bm25))

        merged = self._add_neighbors(merged, neighbor_window=neighbor_window)


 
        scored = []
        for gi in merged:
            # dense score fallback if not in dense set
            base = dense_scores.get(gi, float(np.dot(q_vec, self.embeddings[gi])) if self.embeddings is not None else 0.0)
            bonus = 0.0
          
            scored.append((gi, base + bonus))
        scored.sort(key=lambda x: x[1], reverse=True)

        pool = [gi for gi, _ in scored][: max(top_k * 8, top_k)]

        if allow_rerank and self.use_cross_encoder:
            pool = self._crossencoder_rerank(query, pool)

        final = pool[:top_k]

        # Build your usual result dicts
        results = []
        for rank, gi in enumerate(final):
            c = self.chunks[gi]
            results.append({
                "rank": rank,
                "distance": float("nan"),  # qdrant returns similarity; keep nan or compute proxy
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "title": c.title,
                "chunk_type": c.chunk_type,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            })
        return results

# =========================
# 4. LLM answering (Mistral 7B) + strict formatting enforcement
# =========================

def build_rag_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    max_chunk_chars = 2200

    context_blocks = []
    for r in retrieved:
        paper_tag = f"[Paper_{r['title']}]"
        meta = f"type={r.get('chunk_type','text')} page={r.get('page_start',-1)}"
        header = f"{paper_tag} {r['title']} ({r['paper_id']}) {meta}"
        snippet = (r["text"] or "")[:max_chunk_chars]
        context_blocks.append(f"{header}\n{snippet}")

    context_str = "\n\n---\n\n".join(context_blocks)

    valid_tags = sorted({f"[Paper_{r['title']}]" for r in retrieved})
    valid_tags_str = " ".join(valid_tags)

    system_msg = (
        "You are an expert biomedical research assistant.\n"
        "You MUST ONLY use information that appears in the provided context.\n"
        "If something is not in the context, say the context is insufficient.\n"
        "You must always cite sources using ONLY the provided [Paper_*.pdf] tags.\n"
        "ABSOLUTELY DO NOT output bracketed numeric citations like [12] or [36, 37] or (12).\n"
        "Those are in-paper reference markers and must be removed.\n"
        "FORMAT COMPLIANCE IS CRITICAL.\n"
    )

    user_msg = f"""
Context from retrieved documents:

{context_str}

Question:
{question}

VALID SOURCE TAGS:
{valid_tags_str}

Write your output in EXACTLY TWO SECTIONS, in this order:

FACTS
ANSWER

1) FACTS
- Write 8–20 bullet points.
- Each bullet MUST start with exactly ONE source tag and then a space, like:
  - [Paper_example.pdf] This sentence is a fact supported by that paper.
- Each bullet must be a single factual sentence supported by that source.
- Do NOT invent any tags.
- Do NOT put multiple facts in one bullet.

2) ANSWER
- Write 1–3 short paragraphs answering the question plainly.
- When you use a fact, cite the relevant tag(s) at the end of the sentence.
- Do NOT introduce new facts not present in FACTS.
- If context is insufficient, say so explicitly.
""".strip()

    return (
        f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
        f"{user_msg}\n[/INST]"
    )


def strip_bracket_number_citations(text: str) -> str:
    text = re.sub(
        r"\[\s*\d+\s*(?:[-–]\s*\d+)?\s*(?:,\s*\d+\s*(?:[-–]\s*\d+)?\s*)*\]",
        "",
        text,
    )
    text = re.sub(
        r"\(\s*\d+\s*(?:[-–]\s*\d+)?\s*(?:,\s*\d+\s*(?:[-–]\s*\d+)?\s*)*\)",
        "",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_valid_tags(retrieved: List[Dict[str, Any]]) -> set:
    return {f"[Paper_{r['title']}]" for r in retrieved}


def validate_llm_output(text: str, valid_tags: set) -> Tuple[bool, str]:
    """
    Enforce:
      - has FACTS and ANSWER sections
      - FACTS has 8-20 bullets
      - each bullet starts with exactly one valid tag
      - no bullet contains more than one [Paper_
    """
    if not text:
        return False, "empty output"

    # basic sections
    if "FACTS" not in text or "ANSWER" not in text:
        return False, "missing FACTS/ANSWER headings"

    # extract FACTS block
    m = re.search(r"FACTS\s*:?\s*(.*?)(?:\n\s*ANSWER\s*:|\Z)", text, flags=re.S | re.I)
    if not m:
        return False, "could not parse FACTS block"
    facts_block = m.group(1).strip()

    bullets = [ln.strip() for ln in facts_block.splitlines() if ln.strip().startswith("-")]
    if not (8 <= len(bullets) <= 20):
        return False, f"FACTS bullet count {len(bullets)} not in [8,20]"

    for b in bullets:
        # must start "- [Paper_xxx.pdf] "
        m2 = re.match(r"^-\s+(\[Paper_[^\]]+\])\s+", b)
        if not m2:
            return False, f"bullet does not start with a tag: {b[:80]}"
        tag = m2.group(1)
        if tag not in valid_tags:
            return False, f"invalid tag used: {tag}"
        if b.count("[Paper_") != 1:
            return False, "bullet contains multiple Paper tags"

    return True, "ok"


def generate_answer_with_llm(question: str, retrieved: List[Dict[str, Any]], max_retries: int = 2) -> str:
    if not retrieved:
        return "No relevant chunks were retrieved; I don't have enough information to answer."

    valid_tags = extract_valid_tags(retrieved)

    prompt = build_rag_prompt(question, retrieved)

    for attempt in range(max_retries + 1):
        outputs = llm_pipeline(
            prompt,
            num_return_sequences=1,
            eos_token_id=tokenizer_llm.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
        answer = strip_bracket_number_citations(answer)

        ok, why = validate_llm_output(answer, valid_tags)
        if ok:
            return answer

        # If invalid, append a short corrective instruction and retry
        prompt = prompt + (
            "\n\n[FORMAT FIX]\n"
            "Your previous output violated the required format. "
            "Regenerate from scratch with EXACTLY TWO SECTIONS: FACTS then ANSWER. "
            "FACTS must have 8–20 bullets, each bullet must start with exactly ONE valid [Paper_*.pdf] tag, "
            "and each bullet must contain exactly ONE fact sentence.\n"
            f"VALID TAGS: {' '.join(sorted(valid_tags))}\n"
        )

    return answer  # return last attempt even if imperfect


# =========================
# 5. Main routine
# =========================
def qdrant_collection_exists(client, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False
    



def main():
    import argparse
    import os
    import math
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-parse PDFs and rebuild chunks + Qdrant collection.")
    parser.add_argument("--no_tables", action="store_true",
                        help="Disable table extraction via pdfplumber.")
    parser.add_argument("--max_words", type=int, default=450,
                        help="Max words per sentence-aware chunk (default: 450).")
    parser.add_argument("--overlap_sents", type=int, default=4,
                        help="Sentence overlap between chunks (default: 4).")

    parser.add_argument("--top_k", type=int, default=8,
                        help="Final chunks to pass to LLM (default: 8).")
    parser.add_argument("--dense_fetch_k", type=int, default=80,
                        help="Dense candidates to fetch (default: 80).")
    parser.add_argument("--bm25_fetch_k", type=int, default=80,
                        help="BM25 candidates to fetch (default: 80).")
    parser.add_argument("--neighbor_window", type=int, default=1,
                        help="Also include +/- this many neighbor text chunks (default: 1).")

    parser.add_argument("--no_bm25", action="store_true",
                        help="Disable BM25 even if installed.")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable CrossEncoder reranking (downloads model if not cached).")

    # Qdrant options (Qdrant-only main)
    parser.add_argument("--qdrant_host", default="localhost",
                        help="Qdrant host (default: localhost).")
    parser.add_argument("--qdrant_port", type=int, default=6333,
                        help="Qdrant port (default: 6333).")
    parser.add_argument("--qdrant_collection", default="cmem_chunks",
                        help="Qdrant collection name (default: cmem_chunks).")
    parser.add_argument("--recreate_collection", action="store_true",
                        help="Drop + recreate the Qdrant collection even if it exists (DANGEROUS).")

    # Debugging / verbosity
    parser.add_argument("--debug", action="store_true",
                        help="Enable extra debug logging and stack traces.")
    parser.add_argument("--debug_chars", type=int, default=2200,
                        help="How many characters of each retrieved chunk to print (default: 2200).")
    parser.add_argument("--no_debug_chunks", action="store_true",
                        help="Disable printing retrieved chunk text (still prints metadata).")

    args = parser.parse_args()

    include_tables = not args.no_tables

    # -------------------------
    # 0) Quick sanity checks
    # -------------------------
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(f"PDF_DIR not found: {PDF_DIR}")

    # -------------------------
    # 1) Parse + chunk
    # -------------------------
    try:
        if args.rebuild or not os.path.exists(CHUNKS_PATH):
            print("[main] Parsing PDFs and building chunks...")
            papers = parse_all_pdfs(PDF_DIR, extract_tables=include_tables)
            chunks = build_all_chunks(
                papers,
                max_words=args.max_words,
                overlap_sents=args.overlap_sents,
                include_tables=include_tables
            )
            save_chunks_jsonl(chunks, CHUNKS_PATH)
        else:
            print(f"[main] Found existing chunks at {CHUNKS_PATH}.")
            chunks = load_chunks_jsonl(CHUNKS_PATH)
    except Exception as e:
        print("[error] Failed during parse/chunk stage.")
        if args.debug:
            traceback.print_exc()
        raise

    # -------------------------
    # 2) Qdrant index (Qdrant-only)
    # -------------------------
    try:
        rag = QdrantRagIndex(
            model_name=EMBED_MODEL_NAME,
            device="cpu",
            host=args.qdrant_host,
            port=args.qdrant_port,
            collection=args.qdrant_collection,
            use_bm25=not args.no_bm25,
            use_cross_encoder=args.rerank,
        )

        # Decide whether to (re)build collection
        need_build = bool(args.rebuild or args.recreate_collection)

        if not need_build:
            if not qdrant_collection_exists(rag.client, args.qdrant_collection):
                print("[main] Qdrant collection not found; will build it now.")
                need_build = True

        if need_build:
            why = "rebuild/recreate requested" if (args.rebuild or args.recreate_collection) else "missing collection"
            print(f"[main] Building Qdrant collection ({why})...")
            rag.build_index(chunks, cache_embeddings=True, recreate=True)
        else:
            print("[main] Using existing Qdrant collection.")
            rag.load(chunks)

    except Exception as e:
        print("[error] Failed during Qdrant index stage.")
        if args.debug:
            traceback.print_exc()
        raise

    # -------------------------
    # 3) Interactive QA loop
    # -------------------------
    print("\nQdrant RAG + Mistral 7B ready. Ask questions about your PDFs (type 'exit' to quit).\n")

    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        try:
            print("[main] Retrieving relevant chunks...")
            retrieved = rag.search(
                q,
                top_k=args.top_k,
                dense_fetch_k=args.dense_fetch_k,
                bm25_fetch_k=args.bm25_fetch_k,
                neighbor_window=args.neighbor_window,
                allow_rerank=args.rerank,
            )

            # Debug print (metadata + optional text)
            print("\n[debug] Retrieved chunks:")
            for r in retrieved:
                dist = r.get("distance")
                if isinstance(dist, (int, float)) and not math.isnan(dist):
                    dist_str = f"{dist:.4f}"
                else:
                    dist_str = "NA"

                print(
                    f"--- [{r.get('rank','?')}] chunk_id={r.get('chunk_id')} "
                    f"{r.get('title')} ({r.get('paper_id')}) "
                    f"type={r.get('chunk_type')} page={r.get('page_start')} dist={dist_str}"
                )

                if not args.no_debug_chunks:
                    snippet = (r.get("text") or "")[: args.debug_chars]
                    print(snippet)
                    print()

            print("[main] Calling Mistral 7B...")
            answer = generate_answer_with_llm(q, retrieved, max_retries=2)

            print("\n" + "=" * 80)
            print("ANSWER:\n")
            print(answer)

            print("\nSOURCES USED:")
            for r in retrieved:
                dist = r.get("distance")
                if isinstance(dist, (int, float)) and not math.isnan(dist):
                    dist_str2 = f"{dist:.4f}"
                else:
                    dist_str2 = "NA"
                print(
                    f"[{r.get('rank','?')}] {r.get('title')} ({r.get('paper_id')}) "
                    f"dist={dist_str2} chunk_id={r.get('chunk_id')}"
                )
            print("=" * 80 + "\n")

        except Exception as e:
            print("[error] Failed during query/answer stage.")
            if args.debug:
                traceback.print_exc()
            else:
                print(f"[error] {type(e).__name__}: {e}")
            # continue loop so one bad query doesn't kill the session
            continue


if __name__ == "__main__":
    main()
