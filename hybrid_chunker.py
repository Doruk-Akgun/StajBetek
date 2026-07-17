"""
Hybrid chunking pipeline for corporate / academic documents.

Strategy:
  1. Extract text elements from the PDF along with their font size and position
     (using PyMuPDF's "dict" text extraction, which gives per-span font sizes).
  2. Filter out structural noise (footnotes, bylines, headers/footers) using
     domain-agnostic signals: font size relative to body text, and structural
     heuristics (short lines, no terminal punctuation, high special-char density).
  3. Detect section boundaries using font-size jumps (headings are rendered
     larger than body text in almost every document template).
  4. Within each section, apply RecursiveCharacterTextSplitter to guarantee a
     bounded, predictable chunk size while still respecting paragraph/sentence
     boundaries wherever possible.

This replaces pure semantic chunking with something that is:
  - predictable in chunk size (bounded by chunk_size)
  - aware of real document structure (headings define hard section breaks)
  - free of embedding-outlier artifacts (no acronym/list-heavy sentence spikes)
  - much cheaper (no embedding model needed at all, unless you still want
    noise-filtering quality boosted by it — see USE_EMBEDDING_NOISE_FILTER below)
"""

import os
import re
import fitz  # PyMuPDF
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set to True if you want the embedding-based outlier check from your
# earlier semantic_noise_filter layered on top of the structural checks.
# Set to False to skip loading an embedding model entirely (faster, simpler).
USE_EMBEDDING_NOISE_FILTER = False


# ---------------------------------------------------------------------------
# Step 1: extract blocks with text + font size + position
# ---------------------------------------------------------------------------

def extract_blocks_with_metadata(page):
    """
    Returns a list of dicts: {text, font_size, bbox, is_heading (filled later)}
    Font size per block = size of its largest span (headings are usually
    a single dominant size within the block; using max is more robust than mean
    for blocks that mix a bold heading run with smaller trailing punctuation).
    """
    d = page.get_text("dict")
    blocks = []

    for block in d["blocks"]:
        if "lines" not in block:
            continue  # skip image blocks etc.

        text_parts = []
        max_size = 0
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text += span["text"]
                max_size = max(max_size, span["size"])
            text_parts.append(line_text)

        full_text = "\n".join(text_parts).strip()
        if not full_text:
            continue

        blocks.append({
            "text": full_text,
            "font_size": round(max_size),
            "bbox": block["bbox"],  # (x0, y0, x1, y1)
        })

    return blocks


def get_body_font_size(blocks):
    """Most frequent font size across blocks ≈ body text size."""
    sizes = [b["font_size"] for b in blocks]
    return Counter(sizes).most_common(1)[0][0] if sizes else 10


# ---------------------------------------------------------------------------
# Step 2: domain-agnostic noise filtering (structural, not keyword-based)
# ---------------------------------------------------------------------------

def structural_noise_score(text):
    """Returns 0 (clean) to 1 (likely noise), with no domain-specific keywords."""
    cleaned = text.strip()
    if not cleaned:
        return 1.0
    score = 0.0

    # Real prose ends in terminal punctuation
    if not re.search(r'[.!?]\s*$', cleaned):
        score += 0.3

    # Short, low-density blocks = labels/lists, not prose
    lines = cleaned.split("\n")
    avg_line_len = sum(len(l) for l in lines) / len(lines)
    if avg_line_len < 25 and len(cleaned) < 150:
        score += 0.3

    # High separator density = metadata (bylines, page numbers, IDs)
    special_char_ratio = sum(c in "·|•#—-–_/\\" for c in cleaned) / max(len(cleaned), 1)
    if special_char_ratio > 0.03:
        score += 0.2

    # Very long "words" relative to word count = jammed numbers/codes/dates
    n_words = len(cleaned.split())
    if n_words > 0 and len(cleaned) / n_words > 12:
        score += 0.2

    return min(score, 1.0)


def is_noise_block(block, body_font_size, font_ratio_threshold=0.92,
                    structural_threshold=0.5):
    """
    A block is treated as noise if:
      - its font is noticeably smaller than body text (footnotes, captions), OR
      - its structural score crosses the threshold (short, no punctuation,
        high special-char density, etc.)
    Headings (larger font) are NOT filtered here — they're handled separately
    as section markers in step 3.
    """
    if block["font_size"] < body_font_size * font_ratio_threshold:
        return True
    if structural_noise_score(block["text"]) >= structural_threshold:
        return True
    return False


# ---------------------------------------------------------------------------
# Step 3: heading-aware section splitting
# ---------------------------------------------------------------------------

def split_into_sections(blocks, body_font_size, heading_ratio_threshold=1.15):
    """
    Groups consecutive body blocks into sections, starting a new section
    whenever a block's font size is notably larger than body text (a heading).
    Returns a list of dicts: {heading, text}
    """
    sections = []
    current_heading = None
    current_text_parts = []

    for block in blocks:
        is_heading = block["font_size"] > body_font_size * heading_ratio_threshold

        if is_heading:
            # flush the previous section
            if current_text_parts:
                sections.append({
                    "heading": current_heading,
                    "text": " ".join(current_text_parts).strip()
                })
                current_text_parts = []
            current_heading = block["text"]
        else:
            current_text_parts.append(block["text"].replace("\n", " "))

    # flush the last section
    if current_text_parts:
        sections.append({
            "heading": current_heading,
            "text": " ".join(current_text_parts).strip()
        })

    return sections


# ---------------------------------------------------------------------------
# Step 4: recursive character splitting within each section
# ---------------------------------------------------------------------------

def chunk_sections(sections, chunk_size=500, chunk_overlap=50):
    """
    Applies RecursiveCharacterTextSplitter within each section independently,
    so a chunk never silently crosses a heading boundary. Each chunk keeps
    its section heading as metadata for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    for section in sections:
        if not section["text"]:
            continue
        pieces = splitter.split_text(section["text"])
        for piece in pieces:
            all_chunks.append({
                "heading": section["heading"],
                "content": piece.strip()
            })
    return all_chunks


# ---------------------------------------------------------------------------
# Step 5 (alternative to Step 4): parent-child chunking
# ---------------------------------------------------------------------------
#
# Why: a single flat chunk size is a compromise. Small chunks embed well
# (precise, topically focused -> better retrieval matches) but lack context
# when handed to an LLM. Large chunks give good context but dilute the
# embedding (a 2000-char chunk about three different sub-topics matches
# fewer queries well).
#
# Parent-child chunking resolves this by using two sizes:
#   - PARENT chunks: large, context-rich (e.g. ~1500-2500 chars). Never
#     shown to the embedding model for retrieval -- only returned to the
#     LLM once a child chunk inside them is matched.
#   - CHILD chunks: small (e.g. ~200-400 chars), created by further
#     splitting each parent. These are what actually get embedded and
#     searched against. Each child stores a reference to its parent_id.
#
# At query time: embed the query -> search only over child embeddings ->
# find the best-matching child -> look up its parent_id -> return the
# FULL PARENT TEXT to the LLM as context, not just the small child chunk.
#
# Parents are still built strictly within section boundaries (same as
# Step 4), so a parent never silently crosses a heading boundary either.

def build_parent_child_chunks(sections, parent_chunk_size=2000, parent_overlap=100,
                               child_chunk_size=300, child_overlap=30):
    """
    Returns:
      parents:  dict of {parent_id: {"heading": str, "content": str}}
      children: list of {"child_id": str, "parent_id": str, "heading": str, "content": str}

    Use `children` for building your embedding index (what you search over).
    Use `parents` to resolve a matched child back to its full context
    (what you actually feed to the LLM).
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    parents = {}
    children = []
    parent_counter = 0
    child_counter = 0

    for section in sections:
        if not section["text"]:
            continue

        # a section itself may be longer than one parent chunk -- split it
        parent_pieces = parent_splitter.split_text(section["text"])

        for parent_text in parent_pieces:
            parent_id = f"p{parent_counter}"
            parent_counter += 1
            parents[parent_id] = {
                "heading": section["heading"],
                "content": parent_text.strip()
            }

            # split this parent further into small retrieval-sized children
            child_pieces = child_splitter.split_text(parent_text)
            for child_text in child_pieces:
                child_id = f"c{child_counter}"
                child_counter += 1
                children.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "heading": section["heading"],
                    "content": child_text.strip()
                })

    return parents, _dedupe_children(children)


def get_parent_for_child(child, parents):
    """Convenience lookup: given a matched child dict, return its full parent text."""
    return parents[child["parent_id"]]


def _dedupe_children(children):
    """Drop children with exact-duplicate content (keeps first occurrence)."""
    seen = set()
    deduped = []
    for c in children:
        key = c["content"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Step 5b: parent-child chunking with SEMANTIC child splitting
# ---------------------------------------------------------------------------
#
# Same parent/child architecture as Step 5, but children are found using
# topic-shift detection (embedding distance between sentences) instead of
# a fixed character count.
#
# Why this is more stable than semantic-splitting a whole page/document:
# the semantic splitter now only ever looks at ONE PARENT at a time (e.g.
# ~2000 chars, already confined to a single section). The distance
# distribution it computes percentiles over is local and topically
# homogeneous, so a single acronym-heavy or list-heavy sentence is far
# less likely to dominate the whole distribution and create a spurious
# breakpoint -- which is exactly what caused the orphan "F-score..."
# chunk when splitting semantically over the entire page at once.
#
# Parents still come from the heading-aware structural split, so section
# boundaries (Abstract -> Introduction) are still hard, guaranteed breaks
# -- semantic splitting is only ever refining boundaries *inside* a
# section, never deciding where sections begin or end.

def build_parent_child_chunks_semantic(sections, embedding_model,
                                        parent_chunk_size=2000, parent_overlap=100,
                                        breakpoint_threshold_type="percentile",
                                        breakpoint_threshold_amount=85,
                                        buffer_size=2,
                                        min_child_words=15):
    """
    Returns the same (parents, children) structure as build_parent_child_chunks,
    but children are produced by a SemanticChunker run independently within
    each parent, followed by a short-fragment merge pass (guards against the
    single-outlier-sentence problem you hit earlier).
    """
    from langchain_experimental.text_splitter import SemanticChunker

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    semantic_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        buffer_size=buffer_size,
    )

    parents = {}
    children = []
    parent_counter = 0
    child_counter = 0

    for section in sections:
        if not section["text"]:
            continue

        parent_pieces = parent_splitter.split_text(section["text"])

        for parent_text in parent_pieces:
            parent_id = f"p{parent_counter}"
            parent_counter += 1
            parents[parent_id] = {
                "heading": section["heading"],
                "content": parent_text.strip()
            }

            # semantic split within this single parent only
            semantic_docs = semantic_splitter.create_documents([parent_text])
            child_texts = [d.page_content.strip() for d in semantic_docs if d.page_content.strip()]

            # merge any trailing/short fragments into a neighbor -- guards
            # against the single-sentence-outlier problem seen earlier
            merged = []
            for text in child_texts:
                if merged and len(text.split()) < min_child_words:
                    merged[-1] = merged[-1] + " " + text
                else:
                    merged.append(text)

            for child_text in merged:
                child_id = f"c{child_counter}"
                child_counter += 1
                children.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "heading": section["heading"],
                    "content": child_text.strip()
                })

    return parents, _dedupe_children(children)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _extract_clean_sections(pdf_path, pages="all"):
    """
    Shared preprocessing used by both flat and parent-child chunking:
    extract blocks -> filter noise -> split into heading-bounded sections.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    page_indices = range(len(doc)) if pages == "all" else pages

    all_blocks = []
    for i in page_indices:
        page = doc[i]
        all_blocks.extend(extract_blocks_with_metadata(page))

    if not all_blocks:
        return []

    body_font_size = get_body_font_size(all_blocks)

    # filter noise
    clean_blocks = [b for b in all_blocks if not is_noise_block(b, body_font_size)]

    # optional: layer the embedding-based outlier check on top
    if USE_EMBEDDING_NOISE_FILTER:
        from langchain_huggingface import HuggingFaceEmbeddings
        import numpy as np

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        texts = [b["text"] for b in clean_blocks]
        embeddings = np.array(embedding_model.embed_documents(texts))
        if len(embeddings) > 1:
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            keep = []
            for i, vec in enumerate(embeddings):
                sim = np.dot(vec / np.linalg.norm(vec), centroid)
                if sim < 0.25 and len(texts[i]) < 200:
                    continue  # drop semantic outlier
                keep.append(clean_blocks[i])
            clean_blocks = keep

    sections = split_into_sections(clean_blocks, body_font_size)
    return sections


def extract_and_chunk_pdf(pdf_path, chunk_size=500, chunk_overlap=50, pages="all"):
    """
    Flat chunking (Step 4): one size, one list of chunks with heading metadata.
    """
    sections = _extract_clean_sections(pdf_path, pages=pages)
    return chunk_sections(sections, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def extract_and_chunk_pdf_parent_child(pdf_path, parent_chunk_size=2000,
                                        parent_overlap=100, child_chunk_size=300,
                                        child_overlap=30, pages="all"):
    """
    Parent-child chunking (Step 5): returns (parents, children).
    Children are split by fixed character size (fast, no embedding model needed).
    """
    sections = _extract_clean_sections(pdf_path, pages=pages)
    return build_parent_child_chunks(
        sections,
        parent_chunk_size=parent_chunk_size,
        parent_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_overlap=child_overlap
    )


def extract_and_chunk_pdf_parent_child_semantic(pdf_path, embedding_model,
                                                  parent_chunk_size=2000,
                                                  parent_overlap=100,
                                                  breakpoint_threshold_amount=85,
                                                  buffer_size=2,
                                                  pages="all"):
    """
    Parent-child chunking (Step 5b): returns (parents, children).
    Children are split semantically (topic-shift aware) within each parent,
    instead of by fixed character size. Requires an embedding model.
    """
    sections = _extract_clean_sections(pdf_path, pages=pages)
    return build_parent_child_chunks_semantic(
        sections,
        embedding_model,
        parent_chunk_size=parent_chunk_size,
        parent_overlap=parent_overlap,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        buffer_size=buffer_size,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PDF_DOSYA_ADI = "16.pdf"

    # Switch between the three modes here:
    #   "flat"                 -> heading sections + fixed-size recursive chunks
    #   "parent_child"         -> heading sections as parents + fixed-size children
    #   "parent_child_semantic"-> heading sections as parents + semantically-split children
    MODE = "parent_child_semantic"

    if MODE == "flat":
        print("[1/2] PDF isleniyor (heading-aware + recursive chunking)...")
        try:
            chunks = extract_and_chunk_pdf(
                PDF_DOSYA_ADI, chunk_size=500, chunk_overlap=50, pages="all"
            )
            print(f"\n[2/2] Basarili! Toplam {len(chunks)} chunk bulundu.\n")
            for i, c in enumerate(chunks):
                print(f"================ CHUNK {i+1} (Section: {c['heading']}) ================")
                print(c["content"])
                print("-" * 50 + "\n")
        except FileNotFoundError as e:
            print(f"Hata: {e}")
        except Exception as e:
            print(f"Bir hata olustu: {e}")

    elif MODE == "parent_child":
        print("[1/2] PDF isleniyor (parent-child chunking, fixed-size children)...")
        try:
            parents, children = extract_and_chunk_pdf_parent_child(
                PDF_DOSYA_ADI,
                parent_chunk_size=2000,
                parent_overlap=100,
                child_chunk_size=300,
                child_overlap=30,
                pages="all"
            )
            print(f"\n[2/2] Basarili! {len(parents)} parent, {len(children)} child chunk bulundu.\n")

            for pid, parent in parents.items():
                print(f"================ PARENT {pid} (Section: {parent['heading']}) ================")
                print(parent["content"][:300] + ("..." if len(parent["content"]) > 300 else ""))
                own_children = [c for c in children if c["parent_id"] == pid]
                print(f"  -> {len(own_children)} child chunk(s):")
                for c in own_children:
                    print(f"     [{c['child_id']}] {c['content'][:80]}...")
                print("-" * 50 + "\n")

        except FileNotFoundError as e:
            print(f"Hata: {e}")
        except Exception as e:
            print(f"Bir hata olustu: {e}")

    else:  # parent_child_semantic
        print("[1/3] Embedding modeli yukleniyor...")
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("[2/3] PDF isleniyor (parent-child chunking, semantic children)...")
        try:
            parents, children = extract_and_chunk_pdf_parent_child_semantic(
                PDF_DOSYA_ADI,
                embedding_model,
                parent_chunk_size=2000,
                parent_overlap=100,
                breakpoint_threshold_amount=85,
                buffer_size=2,
                pages="all"
            )
            print(f"\n[3/3] Basarili! {len(parents)} parent, {len(children)} child chunk bulundu.\n")

            for pid, parent in parents.items():
                print(f"================ PARENT {pid} (Section: {parent['heading']}) ================")
                print(parent["content"][:300] + ("..." if len(parent["content"]) > 300 else ""))
                own_children = [c for c in children if c["parent_id"] == pid]
                print(f"  -> {len(own_children)} child chunk(s):")
                for c in own_children:
                    print(f"     [{c['child_id']}] {c['content'][:100]}...")
                print("-" * 50 + "\n")

        except FileNotFoundError as e:
            print(f"Hata: {e}")
        except Exception as e:
            print(f"Bir hata olustu: {e}")
