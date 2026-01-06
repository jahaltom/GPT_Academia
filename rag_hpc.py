#!/usr/bin/env python3
"""
rag_hpc_sections.py  (FULL FIXED VERSION)

Fixes:
Prevents "T h e R e s u l t s ..." spaced-letter garbage by:
   - gap-aware word joining (x0/x1 gap)
   - despacify_text() for spaced-letter runs
   - fallback to page.extract_text() if extract_words output is mostly garbage

Fixes "all sections UNKNOWN" by:
   - despacifying headings before matching (e.g., "R E S U L T S" -> "RESULTS")
   - letting the heading guard peek into the next page (so headings near page bottom still count)
   - keeps your running header/footer blacklist logic

Pipeline:
1) Parse PDFs -> parsed/<paper_id>.json containing pages_blocks + stitched section segments + tables
2) Chunk sentence-aware within each section segment (sections do not bleed)
3) Embed with SentenceTransformer and store in Qdrant
4) Optional BM25 + optional CrossEncoder rerank
5) Interactive Q/A against local Mistral 7B

Usage:
  # Build / rebuild
  python rag_hpc_sections.py --rebuild --recreate_collection --qdrant_host 127.0.0.1 --qdrant_port 6333 --qdrant_collection cmem_chunks

  # Query
  python rag_hpc_sections.py --qdrant_host 127.0.0.1 --qdrant_port 6333 --qdrant_collection cmem_chunks

  # Filter
  python rag_hpc_sections.py --section METHODS --section RESULTS --chunk_type text

Optional deps:
  pip install pdfplumber torch transformers sentence-transformers qdrant-client
  pip install rank-bm25
  (optional) CrossEncoder via sentence-transformers
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import math
import hashlib
from collections import Counter

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

BM25_PATH = os.path.join(INDEX_DIR, "bm25.json")               # tokenized corpus
EMB_CACHE_PATH = os.path.join(INDEX_DIR, "embeddings.npy")     # cache dense embeddings
META_CACHE_PATH = os.path.join(INDEX_DIR, "embeddings_meta.json")  # cache fingerprint

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Embedding model
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Optional CrossEncoder reranker
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Local LLM
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


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
    chunk_type: str      # "text" or "table"
    page_start: int
    page_end: int

    # For neighbor expansion on text chunks
    chunk_index: int = -1

    # MAIN section label ONLY
    section: str = "UNKNOWN"


# =========================
# Reference / boilerplate chunk filter
# =========================

def looks_like_references(text: str) -> bool:
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
# MAIN section heading detection
# =========================

SECTION_CANON = {
    "abstract": "ABSTRACT",
    "summary": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "background": "BACKGROUND",
    "methods": "METHODS",
    "materials and methods": "METHODS",
    "materials & methods": "METHODS",
    "methodology": "METHODS",
    "experimental procedures": "METHODS",
    "results": "RESULTS",
    "discussion": "DISCUSSION",
    "conclusion": "CONCLUSION",
    "conclusions": "CONCLUSION",
    "references": "REFERENCES",
    "bibliography": "REFERENCES",
    "acknowledgments": "ACKNOWLEDGMENTS",
    "acknowledgements": "ACKNOWLEDGMENTS",
    "supplementary information": "SUPPLEMENTARY",
    "supplementary": "SUPPLEMENTARY",
    "appendix": "SUPPLEMENTARY",
    "results and discussion": "RESULTS",
    "results & discussion": "RESULTS",
    "discussion and results": "RESULTS",
    "methods and materials": "METHODS",
    "methods & materials": "METHODS",
}



# 1. / 1) / 1: / 1.2.3 / 2.4) etc
_NUM_PREFIX = re.compile(r"^\s*(\d+(\.\d+)*)([\)\.\:\-])\s+")
# Roman numerals: I. / IV) / X: etc
_ROMAN_PREFIX = re.compile(r"^\s*([IVXLCDM]+)([\)\.\:\-])\s+", re.IGNORECASE)
# Common bullets
_BULLET_PREFIX = re.compile(r"^\s*[\-\u2022\*\•]\s+")

def strip_heading_prefix(s: str) -> str:
    if not s:
        return s
    s = s.strip()

    # remove bullets first
    s = _BULLET_PREFIX.sub("", s)

    # remove numbering / roman numerals
    s2 = _NUM_PREFIX.sub("", s)
    if s2 != s:
        s = s2
    s2 = _ROMAN_PREFIX.sub("", s)
    if s2 != s:
        s = s2

    # normalize punctuation + spaces
    s = s.strip().strip(":").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


# =========================
# Spaced-letter repair
# =========================

def looks_spaced_letter_line(s: str) -> bool:
    """
    Detect 'T h i s   i s' style text.
    If a line has lots of single-letter tokens, it's probably spaced-letter encoding.
    """
    if not s:
        return False
    toks = s.strip().split()
    if len(toks) < 8:
        return False
    single = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    return (single / max(1, len(toks))) >= 0.55


def despacify_text(s: str) -> str:
    """
    Convert spaced-letter runs into normal text.

    Examples:
      'R E S U L T S' -> 'RESULTS'
      'M a t e r i a l s a n d M e t h o d s' -> 'Materials and Methods'

    Collapses runs of >=5 single-letter alphabetic tokens.
    """
    if not s:
        return s

    toks = s.split()
    if len(toks) < 8:
        return s

    out = []
    i = 0
    while i < len(toks):
        t = toks[i]

        if len(t) == 1 and t.isalpha():
            j = i
            letters = []
            while j < len(toks) and len(toks[j]) == 1 and toks[j].isalpha():
                letters.append(toks[j])
                j += 1

            if len(letters) >= 5:
                out.append("".join(letters))
                i = j
                continue

        out.append(t)
        i += 1

    collapsed = " ".join(out)
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    collapsed = re.sub(r"\(\s+", "(", collapsed)
    collapsed = re.sub(r"\s+\)", ")", collapsed)
    collapsed = re.sub(r"\s{2,}", " ", collapsed).strip()
    return collapsed




def is_main_section_heading(line: str) -> Optional[str]:
    """
    Returns canonical MAIN section label if this line looks like a MAIN section heading, else None.
    Handles:
      - "1. Introduction"
      - "2) Materials and Methods"
      - "IV. Results and Discussion"
      - "RESULTS" / "R E S U L T S"
    """
    l = (line or "").strip()
    if not l:
        return None
    if len(l) > 160:
        return None

    # fix spaced letters first
    if looks_spaced_letter_line(l):
        l = despacify_text(l)

    # strip numbering/bullets
    l = strip_heading_prefix(l)

    # drop trailing ":" again after stripping
    l = l.strip().strip(":").strip()

    # lower for canon lookup
    low = l.lower()

    # direct canon match
    if low in SECTION_CANON:
        return SECTION_CANON[low]

    # sometimes headings include extra whitespace/punct
    low2 = re.sub(r"\s*&\s*", " & ", low)
    low2 = re.sub(r"\s{2,}", " ", low2).strip()
    if low2 in SECTION_CANON:
        return SECTION_CANON[low2]

    # OPTIONAL: if your PDFs have "Results and Discussion" but with commas etc
    low3 = re.sub(r"[^\w\s&]", "", low2).strip()
    if low3 in SECTION_CANON:
        return SECTION_CANON[low3]

    return None



# =========================
# Sentence splitting (sentence-aware chunking)
# =========================

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

def split_into_sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    t = re.sub(r"\b(e\.g|i\.e|et al)\.\s", lambda m: m.group(0).replace(". ", "<DOT> "), t, flags=re.I)
    parts = _SENT_SPLIT.split(t)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# =========================
# PDF text extraction helpers  (COLUMN-AWARE)
# =========================

def words_to_line_objs(words: List[Dict[str, Any]], y_tol: float = 2.0) -> List[Dict[str, Any]]:
    """
    Like words_to_lines(), but returns line objects with y position:
      [{"top": float, "text": str}, ...]
    Uses gap-aware join:
      - If next token is very close, concatenate without space
      - Else join with space
    """
    if not words:
        return []

    # bucket by approximate line "top"
    buckets: List[List[Dict[str, Any]]] = []
    for w in sorted(words, key=lambda z: (float(z.get("top", 0.0)), float(z.get("x0", 0.0)))):
        placed = False
        wt = float(w.get("top", 0.0))
        for b in buckets:
            if abs(wt - float(b[0].get("top", 0.0))) <= y_tol:
                b.append(w)
                placed = True
                break
        if not placed:
            buckets.append([w])

    line_objs: List[Dict[str, Any]] = []
    for b in buckets:
        b_sorted = sorted(b, key=lambda z: float(z.get("x0", 0.0)))

        parts: List[str] = []
        prev = None
        for z in b_sorted:
            txt = (z.get("text") or "").strip()
            if not txt:
                continue

            if prev is None:
                parts.append(txt)
            else:
                gap = float(z.get("x0", 0.0)) - float(prev.get("x1", 0.0))
                if gap <= 1.2:
                    parts[-1] = parts[-1] + txt
                else:
                    parts.append(txt)
            prev = z

        line = " ".join(parts).strip()
        if line:
            if looks_spaced_letter_line(line):
                line = despacify_text(line)
            line_objs.append({"top": float(b_sorted[0].get("top", 0.0)), "text": line})

    # Ensure stable top->bottom ordering
    line_objs.sort(key=lambda d: float(d["top"]))
    return line_objs


def words_to_lines(words: List[Dict[str, Any]], y_tol: float = 2.0) -> List[str]:
    """Backward-compatible wrapper returning only text lines."""
    return [d["text"] for d in words_to_line_objs(words, y_tol=y_tol)]


def _detect_two_column_split_x(words: List[Dict[str, Any]], page_width: float) -> Optional[float]:
    """
    Heuristic: find a vertical split (gutter) confirming a 2-column layout.

    We scan candidate split_x in the middle band of the page and choose the split
    that minimizes 'crossing' words (words whose bbox spans across split_x).
    """
    if not words or page_width <= 0:
        return None

    W = float(page_width)
    n = len(words)
    if n < 80:
        # too few words to reliably detect columns
        return None

    # candidate splits in the central region
    lo = 0.35 * W
    hi = 0.65 * W
    candidates = np.linspace(lo, hi, 31)

    # precompute x0/x1/xc for speed
    x0 = np.array([float(w.get("x0", 0.0)) for w in words], dtype=np.float32)
    x1 = np.array([float(w.get("x1", 0.0)) for w in words], dtype=np.float32)
    xc = (x0 + x1) / 2.0

    best = None
    best_score = None

    for sx in candidates:
        # words that cross the split
        crossing = np.sum((x0 < sx) & (x1 > sx))

        # balance: enough words on each side (by center)
        left = np.sum(xc < sx)
        right = np.sum(xc >= sx)

        # density near gutter: count centers within a narrow band
        band = 10.0  # points
        near = np.sum((xc >= (sx - band)) & (xc <= (sx + band)))

        # score: prioritize minimal crossing, then minimal near-gutter density
        score = (crossing * 10_000) + near

        if best_score is None or score < best_score:
            best_score = score
            best = sx

    if best is None:
        return None

    sx = float(best)

    # Validate split quality:
    crossing = sum(1 for w in words if float(w.get("x0", 0.0)) < sx < float(w.get("x1", 0.0)))
    left = sum(1 for w in words if (float(w.get("x0", 0.0)) + float(w.get("x1", 0.0))) / 2.0 < sx)
    right = n - left

    # Require:
    # - very few crossings (true gutter)
    # - both sides have decent content
    if crossing > max(3, int(0.02 * n)):
        return None
    if left < int(0.25 * n) or right < int(0.25 * n):
        return None

    return sx


def extract_body_lines(
    page,
    top_margin: float,
    bottom_margin: float,
) -> List[str]:
    """
    Extract body-only lines.
    Primary: extract_words + gap-aware reconstruction
    NOW: column-aware: detect 2-column and read left column top->bottom, then right column top->bottom.
    Fallback: page.extract_text() if output is mostly spaced-letter / broken.
    """
    H = float(page.height or 0.0)
    W = float(page.width or 0.0)

    words = page.extract_words(extra_attrs=["top", "bottom", "x0", "x1", "size"]) or []
    body_words = [
        w for w in words
        if (float(w.get("top", 0.0)) >= top_margin) and (float(w.get("bottom", 0.0)) <= (H - bottom_margin))
    ]

    # --- NEW: 2-column handling ---
    split_x = _detect_two_column_split_x(body_words, page_width=W)

    if split_x is not None:
        sx = float(split_x)

        spanning = []
        left = []
        right = []

        for w in body_words:
            wx0 = float(w.get("x0", 0.0))
            wx1 = float(w.get("x1", 0.0))
            wxc = (wx0 + wx1) / 2.0

            # words that literally cross the gutter -> spanning (titles/headings/captions)
            if wx0 < sx < wx1:
                spanning.append(w)
            else:
                if wxc < sx:
                    left.append(w)
                else:
                    right.append(w)

        # Build line objs so we can merge spanning into left stream by y
        left_lines = words_to_line_objs(left, y_tol=2.0)
        right_lines = words_to_line_objs(right, y_tol=2.0)
        span_lines = words_to_line_objs(spanning, y_tol=2.0)

        # Merge spanning lines into LEFT stream by y (prevents breaking headings across columns)
        # Keep RIGHT stream separate (prevents left/right interleaving).
        merged_left = sorted(left_lines + span_lines, key=lambda d: float(d["top"]))
        lines = [d["text"] for d in merged_left] + [d["text"] for d in right_lines]

    else:
        # original single-column behavior
        lines = words_to_lines(body_words, y_tol=2.0)

    # If too many lines still look spaced-letter, fallback to extract_text()
    if lines:
        bad = sum(1 for ln in lines if looks_spaced_letter_line(ln))
        if bad / max(1, len(lines)) <= 0.25:
            return lines

    # Fallback
    txt = page.extract_text() or ""
    txt = re.sub(r"\r", "\n", txt)
    raw_lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]

    cleaned: List[str] = []
    for ln in raw_lines:
        if looks_spaced_letter_line(ln):
            ln = despacify_text(ln)
        cleaned.append(ln)

    return cleaned



# =========================
# Section-block building
# =========================

def build_repeated_line_blacklist(pages_lines: List[List[str]], min_frac: float = 0.30) -> set:
    """
    Lines that repeat on many pages are likely running headers/footers.
    We blacklist exact matches.
    """
    counts = Counter()
    n_pages = len(pages_lines)
    for lines in pages_lines:
        uniq = set(ln.strip() for ln in lines if ln and ln.strip())
        for ln in uniq:
            if 3 <= len(ln) <= 140:
                counts[ln] += 1
    thr = max(2, int(math.ceil(min_frac * max(n_pages, 1))))
    return {ln for ln, c in counts.items() if c >= thr}


def split_page_into_section_blocks(
    lines: List[str],
    prev_section: str,
    blacklist: set,
    min_body_after_heading_chars: int = 200,
    next_page_preview_text: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Walk lines top->bottom and split into blocks whenever a MAIN section heading is detected.
    Text before the first heading inherits prev_section.

    Guard: accept a heading only if there is enough body text AFTER it; now also peeks into the
    next page preview so page-bottom headings still count.
    """
    blocks: List[Dict[str, Any]] = []
    cur_section = prev_section if prev_section else "UNKNOWN"
    buf: List[str] = []

    def flush():
        nonlocal buf
        t = "\n".join(buf).strip()
        if t:
            blocks.append({"section": cur_section, "text": t})
        buf = []

    # Pre-filter: remove blacklisted lines; despacify as needed
    filtered: List[str] = []
    for ln in lines:
        ln2 = (ln or "").strip()
        if not ln2:
            continue
        if looks_spaced_letter_line(ln2):
            ln2 = despacify_text(ln2)
        if ln2 in blacklist:
            continue
        filtered.append(ln2)

    i = 0
    while i < len(filtered):
        ln = filtered[i].strip()

        merged = None
        if i + 1 < len(filtered):
            merged = (ln + " " + filtered[i + 1].strip()).strip()

        cand = is_main_section_heading(ln)
        cand_merged = is_main_section_heading(merged) if merged else None

        chosen = None
        used_merged = False
        if cand_merged and not cand:
            chosen = cand_merged
            used_merged = True
        elif cand:
            chosen = cand

        if chosen:
            remaining_lines = filtered[i + (2 if used_merged else 1):]
            remaining_text = " ".join(remaining_lines).strip()
            preview = (next_page_preview_text or "").strip()

            if (len(remaining_text) + len(preview)) < min_body_after_heading_chars:
                buf.append(ln)
                i += 1
                continue

            flush()
            cur_section = chosen
            i += (2 if used_merged else 1)
            continue

        buf.append(ln)
        i += 1

    flush()
    return blocks, cur_section


def build_section_segments(pages_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: list of {"page": i, "blocks": [{"section":..., "text":...}, ...]}
    Output segments across pages: list of {"section", "text", "page_start", "page_end"}
    by stitching consecutive blocks with same section across adjacent pages.
    """
    segments: List[Dict[str, Any]] = []
    for pb in pages_blocks:
        page = int(pb["page"])
        for b in pb["blocks"]:
            sec = (b.get("section") or "UNKNOWN").strip()
            txt = (b.get("text") or "").strip()
            if not txt:
                continue

            if segments and segments[-1]["section"] == sec and segments[-1]["page_end"] == page - 1:
                segments[-1]["text"] += "\n\n" + txt
                segments[-1]["page_end"] = page
            elif segments and segments[-1]["section"] == sec and segments[-1]["page_end"] == page:
                segments[-1]["text"] += "\n\n" + txt
            else:
                segments.append({
                    "section": sec,
                    "text": txt,
                    "page_start": page,
                    "page_end": page,
                })
    return segments


# =========================
# Table extraction
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


# =========================
# Parse PDFs -> parsed/<paper_id>.json
# =========================

def parse_pdf_to_structured(
    pdf_path: str,
    extract_tables: bool = True,
    top_margin: float = 60.0,
    bottom_margin: float = 60.0,
    blacklist_min_frac: float = 0.30,
    min_body_after_heading_chars: int = 200,
) -> Dict[str, Any]:
    """
    Produces:
      - pages_lines (body-only)
      - pages_blocks: per page section blocks
      - segments: stitched section segments across pages
      - tables: extracted tables with page index
    """
    pages_lines: List[List[str]] = []
    tables: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            lines = extract_body_lines(page, top_margin=top_margin, bottom_margin=bottom_margin)
            pages_lines.append(lines)

            if extract_tables:
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []
                for t_idx, t in enumerate(raw_tables):
                    md = _table_to_markdown(t)
                    if md.strip():
                        tables.append({"page": i, "table_index": t_idx, "markdown": md})

    blacklist = build_repeated_line_blacklist(pages_lines, min_frac=blacklist_min_frac)

    prev_sec = "UNKNOWN"
    pages_blocks: List[Dict[str, Any]] = []
    for i, lines in enumerate(pages_lines):
        next_preview = ""
        if i + 1 < len(pages_lines):
            next_preview = " ".join(pages_lines[i + 1][:20])

        blocks, prev_sec = split_page_into_section_blocks(
            lines=lines,
            prev_section=prev_sec,
            blacklist=blacklist,
            min_body_after_heading_chars=min_body_after_heading_chars,
            next_page_preview_text=next_preview,
        )
        pages_blocks.append({"page": i, "blocks": blocks})

    segments = build_section_segments(pages_blocks)
    full_text = "\n\n".join("\n".join(lines) for lines in pages_lines)

    return {
        "full_text": full_text,
        "pages_lines": pages_lines,
        "pages_blocks": pages_blocks,
        "segments": segments,
        "tables": tables,
        "blacklist_size": len(blacklist),
    }


def parse_all_pdfs(
    pdf_dir: str,
    extract_tables: bool = True,
    top_margin: float = 60.0,
    bottom_margin: float = 60.0,
    blacklist_min_frac: float = 0.30,
    min_body_after_heading_chars: int = 200,
) -> List[Paper]:
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    papers: List[Paper] = []
    for fname in sorted(pdf_files):
        pdf_path = os.path.join(pdf_dir, fname)
        print(f"[parse] Parsing {pdf_path} ...")

        parsed = parse_pdf_to_structured(
            pdf_path=pdf_path,
            extract_tables=extract_tables,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            blacklist_min_frac=blacklist_min_frac,
            min_body_after_heading_chars=min_body_after_heading_chars,
        )

        title = os.path.basename(pdf_path)
        paper_id = os.path.splitext(title)[0]
        paper = Paper(paper_id=paper_id, title=title, text=parsed["full_text"])

        parsed_out = os.path.join(PARSED_DIR, f"{paper.paper_id}.json")
        with open(parsed_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **asdict(paper),
                    "pages_blocks": parsed["pages_blocks"],
                    "segments": parsed["segments"],
                    "tables": parsed["tables"],
                    "blacklist_size": parsed["blacklist_size"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        papers.append(paper)

    print(f"[parse] Parsed {len(papers)} papers.")
    return papers


def load_parsed_json(paper_id: str) -> Dict[str, Any]:
    parsed_out = os.path.join(PARSED_DIR, f"{paper_id}.json")
    if not os.path.exists(parsed_out):
        return {}
    with open(parsed_out, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Chunking by SECTION SEGMENTS (sentence-aware)
# =========================

def chunk_section_segments(
    paper: Paper,
    max_words: int = 450,
    overlap_sents: int = 4,
) -> List[Chunk]:
    data = load_parsed_json(paper.paper_id)
    segments = data.get("segments", []) or []
    if not segments:
        return []

    chunks: List[Chunk] = []
    chunk_idx = 0

    for seg in segments:
        section = (seg.get("section") or "UNKNOWN").strip()
        page_start = int(seg.get("page_start", -1))
        page_end = int(seg.get("page_end", -1))
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue

        # Safety: if any spaced-letter survived, clean it once more.
        if looks_spaced_letter_line(txt[:500]):
            txt = despacify_text(txt)

        sents = split_into_sentences(txt)
        if not sents:
            continue

        sent_words = [len(s.split()) for s in sents]
        i = 0
        while i < len(sents):
            w = 0
            j = i
            buf: List[str] = []

            while j < len(sents) and (w + sent_words[j]) <= max_words:
                buf.append(sents[j])
                w += sent_words[j]
                j += 1

            if not buf:
                buf = [sents[i]]
                j = i + 1

            text = " ".join(buf).strip()

            drop = (section == "REFERENCES") and looks_like_references(text)
            if text and not drop:

                chunks.append(
                    Chunk(
                        chunk_id=f"{paper.paper_id}_textchunk_{chunk_idx}",
                        paper_id=paper.paper_id,
                        title=paper.title,
                        text=text,
                        chunk_type="text",
                        page_start=page_start,
                        page_end=page_end,
                        chunk_index=chunk_idx,
                        section=section,
                    )
                )
                chunk_idx += 1

            i = max(j - overlap_sents, i + 1)

    return chunks


def chunk_tables(
    paper: Paper,
    inherit_section_from_page: bool = True,
) -> List[Chunk]:
    data = load_parsed_json(paper.paper_id)
    tables = data.get("tables", []) or []
    segments = data.get("segments", []) or []

    page_to_section: Dict[int, str] = {}
    if inherit_section_from_page:
        for seg in segments:
            sec = (seg.get("section") or "UNKNOWN").strip()
            ps = int(seg.get("page_start", -1))
            pe = int(seg.get("page_end", -1))
            for p in range(ps, pe + 1):
                page_to_section[p] = sec

    chunks: List[Chunk] = []
    for i, t in enumerate(tables):
        md = (t.get("markdown") or "").strip()
        if not md:
            continue
        page = int(t.get("page", -1))
        sec = page_to_section.get(page, "UNKNOWN") if inherit_section_from_page else "UNKNOWN"
        chunks.append(
            Chunk(
                chunk_id=f"{paper.paper_id}_table_{i}",
                paper_id=paper.paper_id,
                title=paper.title,
                text=md,
                chunk_type="table",
                page_start=page,
                page_end=page,
                chunk_index=-1,
                section=sec,
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
        all_chunks.extend(chunk_section_segments(p, max_words=max_words, overlap_sents=overlap_sents))
        if include_tables:
            all_chunks.extend(chunk_tables(p, inherit_section_from_page=True))
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

            if "chunk_index" not in data:
                ci = -1
                if data.get("chunk_type") == "text":
                    m = re.search(r"_textchunk_(\d+)$", data.get("chunk_id", ""))
                    if m:
                        ci = int(m.group(1))
                data["chunk_index"] = ci

            if "section" not in data:
                data["section"] = "UNKNOWN"

            if "page_start" not in data:
                data["page_start"] = -1
            if "page_end" not in data:
                data["page_end"] = -1

            chunks.append(Chunk(**data))

    print(f"[chunk] Loaded {len(chunks)} chunks from {path}")
    return chunks


# =========================
# Index: Dense + optional BM25 + optional CrossEncoder rerank
# =========================

_word_tok = re.compile(r"[A-Za-z0-9_+\-Δ]+")  # keep some science tokens

def bm25_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_tok.findall(text or "")]


def corpus_fingerprint(chunks: List[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update((c.chunk_id + "\n").encode("utf-8"))
        h.update((c.paper_id + "\n").encode("utf-8"))
        h.update((c.chunk_type + "\n").encode("utf-8"))
        h.update((str(c.page_start) + "\n").encode("utf-8"))
        h.update((str(c.page_end) + "\n").encode("utf-8"))
        h.update((c.section + "\n").encode("utf-8"))
        h.update((str(len(c.text)) + "\n").encode("utf-8"))
        h.update((c.text[:200] + "\n").encode("utf-8", errors="ignore"))
    return h.hexdigest()


class QdrantRagIndex:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        host: str = "localhost",
        port: int = 6333,
        collection: str = "cmem_chunks",
        use_bm25: bool = True,
        use_cross_encoder: bool = False,
        store_text_in_qdrant: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        print(f"[embed] Loading SentenceTransformer: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.client = QdrantClient(host=host, port=port, timeout=120.0)
        self.collection = collection

        self.use_bm25 = bool(use_bm25 and HAS_BM25)
        self.use_cross_encoder = bool(use_cross_encoder and HAS_CE)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME) if self.use_cross_encoder else None
        self.store_text_in_qdrant = bool(store_text_in_qdrant)

        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

        self.bm25 = None
        self.bm25_tokens: List[List[str]] = []

        # neighbor lookup for text chunks (paper_id, chunk_index)->global_index
        self.text_lookup: Dict[Tuple[str, int], int] = {}

    def _build_text_lookup(self):
        self.text_lookup = {}
        for gi, c in enumerate(self.chunks):
            if c.chunk_type == "text" and c.chunk_index >= 0:
                self.text_lookup[(c.paper_id, c.chunk_index)] = gi

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def recreate_collection(self, dim: int) -> None:
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    @staticmethod
    def _build_min_should(should_conditions: List[qm.FieldCondition], min_count: int = 1):
        try:
            return qm.MinShould(conditions=should_conditions, min_count=min_count)
        except Exception:
            return {"conditions": should_conditions, "min_count": min_count}

    def _qdrant_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[qm.Filter]:
        """
        Supports:
          - exact AND filters: {"paper_id": "...", "chunk_type": "text"}
          - OR list for section: {"section__in": ["METHODS","RESULTS"]}
          - OR list for paper_id: {"paper_id__in": ["msaf211","foo"]}
        """
        if not filter_dict:
            return None

        must: List[qm.FieldCondition] = []
        should: List[qm.FieldCondition] = []
        need_min_should = False

        for k, v in filter_dict.items():
            if k == "section__in":
                for sec in v:
                    should.append(qm.FieldCondition(key="section", match=qm.MatchValue(value=sec)))
                need_min_should = True
            elif k == "paper_id__in":
                for pid in v:
                    should.append(qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=pid)))
                need_min_should = True
            else:
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))

        if should and need_min_should:
            ms = self._build_min_should(should, min_count=1)
            try:
                return qm.Filter(must=must or None, min_should=ms)
            except Exception:
                return qm.Filter(must=must or None, should=should, min_should=ms)

        if should:
            return qm.Filter(must=must or None, should=should)

        return qm.Filter(must=must or None)

    def build_index(self, chunks: List[Chunk], cache_embeddings: bool = True, recreate: bool = True) -> None:
        self.chunks = chunks
        self._build_text_lookup()

        texts = [c.text for c in chunks]
        fp = corpus_fingerprint(chunks)

        print(f"[index] Embedding {len(texts)} chunks with {self.model_name} (normalized)...")
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")
        self.embeddings = embeddings

        if cache_embeddings:
            np.save(EMB_CACHE_PATH, embeddings)
            with open(META_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"fingerprint": fp, "n": len(chunks)}, f)

        dim = int(embeddings.shape[1])

        if recreate:
            print(f"[qdrant] Recreating collection '{self.collection}' (dim={dim})...")
            self.recreate_collection(dim)

        print("[qdrant] Upserting points in batches...")
        BATCH = 256
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
                    "section": c.section,
                }
                if self.store_text_in_qdrant:
                    payload["text"] = c.text
                points.append(qm.PointStruct(id=i, vector=embeddings[i].tolist(), payload=payload))

            self.client.upsert(collection_name=self.collection, points=points, wait=True)

            if (start // BATCH) % 10 == 0:
                print(f"[qdrant] Upserted {end}/{len(chunks)} points...")

        if self.use_bm25:
            print("[bm25] Building BM25 corpus...")
            self.bm25_tokens = [bm25_tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(self.bm25_tokens)
            with open(BM25_PATH, "w", encoding="utf-8") as f:
                json.dump({"tokens": self.bm25_tokens}, f)
            print("[bm25] BM25 built.")

        print("[qdrant] Index ready.")

    def load(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self._build_text_lookup()

        fp = corpus_fingerprint(chunks)
        cache_ok = False
        if os.path.exists(EMB_CACHE_PATH) and os.path.exists(META_CACHE_PATH):
            try:
                with open(META_CACHE_PATH, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                if meta.get("fingerprint") == fp and int(meta.get("n", -1)) == len(chunks):
                    cache_ok = True
            except Exception:
                cache_ok = False

        if cache_ok:
            self.embeddings = np.load(EMB_CACHE_PATH).astype("float32")
        else:
            print("[warn] Embedding cache missing/stale; re-embedding now.")
            texts = [c.text for c in self.chunks]
            self.embeddings = self.model.encode(
                texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            ).astype("float32")
            np.save(EMB_CACHE_PATH, self.embeddings)
            with open(META_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"fingerprint": fp, "n": len(chunks)}, f)

        if self.use_bm25 and os.path.exists(BM25_PATH):
            with open(BM25_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.bm25_tokens = data.get("tokens", [])
            if self.bm25_tokens:
                self.bm25 = BM25Okapi(self.bm25_tokens)

        print(f"[qdrant] Loaded chunks ({len(self.chunks)}).")

    def _bm25_candidates(self, query: str, fetch_k: int) -> List[int]:
        if not self.use_bm25 or self.bm25 is None:
            return []
        qtok = bm25_tokenize(query)
        scores = self.bm25.get_scores(qtok)
        idxs = np.argsort(scores)[::-1][:fetch_k]
        return [int(i) for i in idxs if scores[i] > 0]

    def _add_neighbors(self, idxs: List[int], neighbor_window: int = 2) -> List[int]:
        out = set(idxs)
        for gi in idxs:
            c = self.chunks[gi]
            if c.chunk_type != "text" or c.chunk_index < 0:
                continue
    
            base_section = (c.section or "UNKNOWN")
    
            for d in range(1, neighbor_window + 1):
                for ni in (c.chunk_index - d, c.chunk_index + d):
                    gni = self.text_lookup.get((c.paper_id, ni))
                    if gni is None:
                        continue
    
                    # --- NEW: prevent section bleed ---
                    nc = self.chunks[gni]
                    if (nc.section or "UNKNOWN") != base_section:
                        continue
                    # ----------------------------------
    
                    out.add(gni)
        return list(out)
    

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        candidate_indices: List[int],
        k: int,
        lambda_mult: float = 0.75,
    ) -> List[int]:
        if self.embeddings is None or not candidate_indices:
            return candidate_indices[:k]

        selected: List[int] = []
        candidates = candidate_indices.copy()
        relevance = {idx: float(np.dot(query_vec, self.embeddings[idx])) for idx in candidates}

        while candidates and len(selected) < k:
            best_idx = None
            best_score = -1e18

            for idx in candidates:
                if not selected:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(float(np.dot(self.embeddings[idx], self.embeddings[s])) for s in selected)
                score = lambda_mult * relevance[idx] - (1.0 - lambda_mult) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def _crossencoder_rerank(self, query: str, idxs: List[int]) -> List[int]:
        if not self.use_cross_encoder or self.cross_encoder is None:
            return idxs
        pairs = [(query, self.chunks[i].text[:2000]) for i in idxs]
        scores = self.cross_encoder.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [idxs[int(i)] for i in order]

    def search(
        self,
        query: str,
        top_k: int = 12,
        dense_fetch_k: int = 250,
        bm25_fetch_k: int = 250,
        neighbor_window: int = 2,
        allow_rerank: bool = True,
        meta_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        q_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        qfilter = self._qdrant_filter(meta_filter)

        limit = min(dense_fetch_k, len(self.chunks))

        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection,
                query=q_vec.tolist(),
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
        if not merged:
            return []

        scored = []
        for gi in merged:
            base = dense_scores.get(
                gi,
                float(np.dot(q_vec, self.embeddings[gi])) if self.embeddings is not None else 0.0
            )
            scored.append((gi, base))
        scored.sort(key=lambda x: x[1], reverse=True)

        pool = [gi for gi, _ in scored][: max(top_k * 25, 150)]

        if allow_rerank and self.use_cross_encoder:
            pool = self._crossencoder_rerank(query, pool)

        final = self._mmr_select(query_vec=q_vec, candidate_indices=pool, k=top_k, lambda_mult=0.7)

        results = []
        for rank, gi in enumerate(final):
            c = self.chunks[gi]
            results.append({
                "rank": rank,
                "score": float(dense_scores.get(gi, np.dot(q_vec, self.embeddings[gi]))),
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "title": c.title,
                "chunk_type": c.chunk_type,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "text": c.text,
            })
        return results


# =========================
# LLM (lazy load) + basic prompt
# =========================

class LazyLLM:
    def __init__(self, model_name: str = MISTRAL_MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._pipe = None

    def ensure_loaded(self):
        if self._pipe is not None:
            return
        print("[llm] Loading Mistral 7B model...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=1536,
            do_sample=False,
        )
        print("[llm] Mistral 7B loaded.")

    @property
    def tokenizer(self):
        self.ensure_loaded()
        return self._tokenizer

    @property
    def pipe(self):
        self.ensure_loaded()
        return self._pipe


def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    max_chunk_chars = 1800
    context_blocks = []
    for r in retrieved:
        tag = f"[Paper_{r['title']}]"
        meta = f"type={r['chunk_type']} pages={r['page_start']}-{r['page_end']} section={r['section']}"
        header = f"{tag} {meta}"
        snippet = (r["text"] or "")[:max_chunk_chars]
        context_blocks.append(f"{header}\n{snippet}")

    ctx = "\n\n---\n\n".join(context_blocks)
    valid_tags = sorted({f"[Paper_{r['title']}]" for r in retrieved})
    valid_tags_str = " ".join(valid_tags)

    system_msg = (
        "You are an expert biomedical research assistant.\n"
        "Only use information present in the provided context.\n"
        "If the answer is not in the context, say the context is insufficient.\n"
        "Cite sources using ONLY the provided [Paper_*.pdf] tags.\n"
        "Do not output numeric citations like [12] or (12).\n"
        "Do NOT include a References section.\n"
    )

    user_msg = f"""
Context:

{ctx}

Question:
{question}

VALID SOURCE TAGS:
{valid_tags_str}

Answer in 1–3 short paragraphs. Cite tags at end of sentences when used.

CRITICAL CITATION RULE:
- EVERY sentence MUST end with at least one valid source tag from VALID SOURCE TAGS.
- Do NOT add a References section or a tag dump at the end.
- If the context is insufficient, say so explicitly.
""".strip()

    return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg}\n[/INST]"


def strip_bracket_number_citations(text: str) -> str:
    text = re.sub(r"\[\s*\d+\s*(?:[-–]\s*\d+)?\s*(?:,\s*\d+\s*(?:[-–]\s*\d+)?\s*)*\]", "", text)
    text = re.sub(r"\(\s*\d+\s*(?:[-–]\s*\d+)?\s*(?:,\s*\d+\s*(?:[-–]\s*\d+)?\s*)*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def answer_with_llm(llm: LazyLLM, question: str, retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return "No relevant chunks were retrieved; I don't have enough information to answer."
    prompt = build_prompt(question, retrieved)
    outputs = llm.pipe(
        prompt,
        num_return_sequences=1,
        eos_token_id=llm.tokenizer.eos_token_id,
    )
    generated = outputs[0]["generated_text"]
    ans = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
    return strip_bracket_number_citations(ans)


# =========================
# Main
# =========================

def main():
    import traceback

    parser = argparse.ArgumentParser()

    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-parse PDFs and rebuild chunks + Qdrant collection.")
    parser.add_argument("--no_tables", action="store_true",
                        help="Disable table extraction via pdfplumber.")

    # Body region cropping (header/footer suppression)
    parser.add_argument("--top_margin", type=float, default=60.0,
                        help="Ignore words above this many points from top (default: 60).")
    parser.add_argument("--bottom_margin", type=float, default=60.0,
                        help="Ignore words below this many points from bottom (default: 60).")

    # Header blacklist settings
    parser.add_argument("--blacklist_min_frac", type=float, default=0.30,
                        help="Lines repeating on >= this fraction of pages are blacklisted (default: 0.30).")

    # Heading guard
    parser.add_argument("--min_body_after_heading_chars", type=int, default=200,
                        help="Reject headings if too little body text after (default: 200). "
                             "Try 80–120 if you still see too many UNKNOWNs.")

    # Chunking
    parser.add_argument("--max_words", type=int, default=450,
                        help="Max words per sentence-aware chunk (default: 450).")
    parser.add_argument("--overlap_sents", type=int, default=4,
                        help="Sentence overlap between chunks (default: 4).")

    # Retrieval knobs
    parser.add_argument("--top_k", type=int, default=8,
                        help="Final chunks to pass to LLM (default: 8).")
    parser.add_argument("--dense_fetch_k", type=int, default=120,
                        help="Dense candidates to fetch (default: 120).")
    parser.add_argument("--bm25_fetch_k", type=int, default=120,
                        help="BM25 candidates to fetch (default: 120).")
    parser.add_argument("--neighbor_window", type=int, default=1,
                        help="Also include +/- this many neighbor text chunks (default: 1).")

    parser.add_argument("--no_bm25", action="store_true",
                        help="Disable BM25 even if installed.")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable CrossEncoder reranking (downloads model if not cached).")

    # Qdrant
    parser.add_argument("--qdrant_host", default="localhost",
                        help="Qdrant host (default: localhost).")
    parser.add_argument("--qdrant_port", type=int, default=6333,
                        help="Qdrant port (default: 6333).")
    parser.add_argument("--qdrant_collection", default="cmem_chunks",
                        help="Qdrant collection name (default: cmem_chunks).")
    parser.add_argument("--recreate_collection", action="store_true",
                        help="Drop + recreate the Qdrant collection even if it exists.")
    parser.add_argument("--store_text_in_qdrant", action="store_true",
                        help="Store full chunk text in Qdrant payload (default: OFF).")

    # Filters
    parser.add_argument("--section", action="append", default=[],
                        help="Restrict retrieval to MAIN section labels (repeatable). Example: --section METHODS --section RESULTS")
    parser.add_argument("--paper_id", action="append", default=[],
                        help="Restrict retrieval to these paper_id values (repeatable). Example: --paper_id msaf211")
    parser.add_argument("--chunk_type", default=None, choices=["text", "table"],
                        help="Restrict retrieval to one chunk type (text or table).")

    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Enable extra debug logging and stack traces.")
    parser.add_argument("--debug_chars", type=int, default=2000,
                        help="How many characters of each retrieved chunk to print (default: 2000).")
    parser.add_argument("--no_debug_chunks", action="store_true",
                        help="Disable printing retrieved chunk text (still prints metadata).")

    args = parser.parse_args()

    include_tables = not args.no_tables

    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(f"PDF_DIR not found: {PDF_DIR}")

    # -------------------------
    # 0) Early Qdrant connectivity check
    # -------------------------
    try:
        qc = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=5.0)
        _ = qc.get_collections()
    except Exception as e:
        print("[error] Qdrant is not reachable (connection refused / not running).")
        print(f"[error] host={args.qdrant_host} port={args.qdrant_port}")
        print(f"[error] {type(e).__name__}: {e}")
        print("Fix: start qdrant, or use the correct host/port (e.g. ssh tunnel).")
        raise SystemExit(2)

    # -------------------------
    # 1) Parse + chunk
    # -------------------------
    try:
        if args.rebuild or not os.path.exists(CHUNKS_PATH):
            print("[main] Parsing PDFs and building section segments + chunks...")
            papers = parse_all_pdfs(
                PDF_DIR,
                extract_tables=include_tables,
                top_margin=args.top_margin,
                bottom_margin=args.bottom_margin,
                blacklist_min_frac=args.blacklist_min_frac,
                min_body_after_heading_chars=args.min_body_after_heading_chars,
            )
            chunks = build_all_chunks(
                papers,
                max_words=args.max_words,
                overlap_sents=args.overlap_sents,
                include_tables=include_tables,
            )
            save_chunks_jsonl(chunks, CHUNKS_PATH)
        else:
            print(f"[main] Found existing chunks at {CHUNKS_PATH}.")
            chunks = load_chunks_jsonl(CHUNKS_PATH)

        if args.debug:
            secs = sorted({c.section for c in chunks})
            print("\n[debug] Unique MAIN section labels found in chunks:")
            for s in secs:
                print(" ", s)
            unk = sum(1 for c in chunks if c.section == "UNKNOWN")
            print(f"[debug] UNKNOWN sections: {unk}/{len(chunks)} ({(unk/len(chunks))*100:.2f}%)\n")

    except Exception:
        print("[error] Failed during parse/chunk stage.")
        if args.debug:
            traceback.print_exc()
        raise

    # -------------------------
    # 2) Qdrant index
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
            store_text_in_qdrant=args.store_text_in_qdrant,
        )

        need_build = bool(args.rebuild or args.recreate_collection)
        if not need_build and not rag.collection_exists():
            print("[main] Qdrant collection not found; will build it now.")
            need_build = True

        if need_build:
            why = "rebuild/recreate requested" if (args.rebuild or args.recreate_collection) else "missing collection"
            print(f"[main] Building Qdrant collection ({why})...")
            rag.build_index(chunks, cache_embeddings=True, recreate=True)
        else:
            print("[main] Using existing Qdrant collection.")
            rag.load(chunks)

    except Exception:
        print("[error] Failed during Qdrant index stage.")
        if args.debug:
            traceback.print_exc()
        raise

    # -------------------------
    # 3) Interactive QA loop
    # -------------------------
    llm = LazyLLM(MISTRAL_MODEL_NAME)
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

            meta_filter: Dict[str, Any] = {}
            if args.chunk_type:
                meta_filter["chunk_type"] = args.chunk_type

            if args.section:
                meta_filter["section__in"] = [s.strip().upper() for s in args.section if s.strip()]

            if args.paper_id:
                meta_filter["paper_id__in"] = [p.strip() for p in args.paper_id if p.strip()]

            retrieved = rag.search(
                q,
                top_k=args.top_k,
                dense_fetch_k=args.dense_fetch_k,
                bm25_fetch_k=args.bm25_fetch_k,
                neighbor_window=args.neighbor_window,
                allow_rerank=args.rerank,
                meta_filter=meta_filter if meta_filter else None,
            )

            print("\n[debug] Retrieved chunks:")
            for r in retrieved:
                score = r.get("score", float("nan"))
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) and not math.isnan(score) else "NA"
                print(
                    f"--- [{r.get('rank','?')}] chunk_id={r.get('chunk_id')} "
                    f"{r.get('title')} ({r.get('paper_id')}) "
                    f"type={r.get('chunk_type')} pages={r.get('page_start')}-{r.get('page_end')} "
                    f"section={r.get('section','UNKNOWN')} score={score_str}"
                )
                if not args.no_debug_chunks:
                    snippet = (r.get("text") or "")[: args.debug_chars]
                    print(snippet)
                    print()

            print("[main] Calling Mistral 7B...")
            ans = answer_with_llm(llm, q, retrieved)

            print("\n" + "=" * 80)
            print("ANSWER:\n")
            print(ans)
            print("=" * 80 + "\n")

        except Exception as e:
            print("[error] Failed during query/answer stage.")
            if args.debug:
                traceback.print_exc()
            else:
                print(f"[error] {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    main()
