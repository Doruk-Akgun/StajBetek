"""
Script that reads a PDF, splits it into chunks, embeds each chunk using
LM Studio, stores everything in a Chroma DB, and runs a RAG-based
question-answering query on top of it.

Chunking strategy: HYBRID = structural pre-split (paragraphs / headings)
+ semantic split (embedding-distance breakpoints) within each structural
unit + size-based merge/split fallback.

Retrieval strategy: HyDE (Hypothetical Document Embeddings).
  Instead of embedding the raw user question, the LLM first generates a
  short hypothetical passage that *would* answer the question, written in
  the same register as the source documents. That hypothetical passage is
  embedded and used for the Chroma similarity search — closing the
  vocabulary/register gap between casual questions and formal paper text.
  Set HYDE_ENABLED = False to fall back to plain question-embedding
  retrieval for A/B comparison.
"""

import json
import os
import re
import chromadb
import numpy as np
import requests
from pypdf import PdfReader


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
PDF_PATH = "20.pdf"
CHROMA_DB_PATH = "./experimental_rag_db"
COLLECTION_NAME = "academic_paper_20_1"

# --- Semantic chunking settings ---
SEMANTIC_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT_PERCENTILE = 90
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1800
FALLBACK_OVERLAP = 100

# --- Hybrid / structural chunking settings ---
STRUCTURAL_UNIT_SEMANTIC_THRESHOLD = 400
HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6}\s+.+|\d{1,2}(\.\d{1,2})*\.?\s+[A-Z].{0,80}|[A-Z][A-Z\s\-:]{4,80})\s*$"
)

# --- HyDE settings ---
HYDE_ENABLED = True     # flip to False to compare against plain question retrieval
N_RESULTS = 4            # chunks retrieved for the final answer context


# ---------------------------------------------------------
# 1. LM Studio embedding function
# ---------------------------------------------------------
def get_embedding(text):
    url = f"{LM_STUDIO_BASE_URL}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"input": text, "model": "local"}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(f"LM Studio Error: {response.status_code} - {response.text}")
    return response.json()["data"][0]["embedding"]


# ---------------------------------------------------------
# 2. LM Studio chat/completion function
# ---------------------------------------------------------
def ask_llm(system_prompt, user_prompt):
    url = f"{LM_STUDIO_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(f"LM Studio Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------
# 2b. HyDE: generate a hypothetical answer passage for retrieval
# ---------------------------------------------------------
def generate_hypothetical_document(question):
    """
    Asks the LLM to write a short, confident-sounding passage that would
    answer the question, in the style of an academic paper. This is NOT
    shown to the user and is NOT used as the final answer — it exists
    purely to be embedded and used as the retrieval query, since it's
    closer in vocabulary/register to the real source chunks than the
    raw question is.
    """
    system_prompt = (
        "You write short hypothetical excerpts from academic papers. "
        "Given a question, write a brief passage (3-5 sentences) in the "
        "formal, technical style of an academic paper that WOULD answer "
        "this question, as if it were extracted directly from such a paper. "
        "Do not hedge, do not say you don't know, do not mention that this "
        "is hypothetical. Just write the passage."
    )
    ans = ask_llm(system_prompt=system_prompt, user_prompt=question)
    print("Hypothetical document:")
    print(ans)
    return ans


# ---------------------------------------------------------
# 3. Extract text from PDF (page by page, keeping page numbers)
# ---------------------------------------------------------
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i + 1, "text": text})
    return pages


# ---------------------------------------------------------
# 4. HYBRID chunking
# ---------------------------------------------------------
def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1 - sim


def split_oversized_chunk(text, max_size, overlap):
    pieces = []
    i = 0
    while i < len(text):
        piece = text[i : i + max_size].strip()
        if piece:
            pieces.append(piece)
        i += max_size - overlap
    return pieces


def split_structural_units(text):
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    units = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        first_line = para.split("\n", 1)[0].strip()
        is_heading = bool(HEADING_PATTERN.match(first_line))
        units.append({"text": para, "is_heading_start": is_heading})
    return units


def semantic_split_unit(text, embed_fn, buffer_size=SEMANTIC_BUFFER_SIZE,
                         breakpoint_percentile=SEMANTIC_BREAKPOINT_PERCENTILE):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    combined = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        combined.append(" ".join(sentences[start:end]))

    embeddings = [embed_fn(c) for c in combined]

    distances = [
        cosine_distance(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]
    if not distances:
        return [text]

    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    raw_chunks = []
    start_idx = 0
    for bp in breakpoints:
        raw_chunks.append(" ".join(sentences[start_idx : bp + 1]))
        start_idx = bp + 1
    raw_chunks.append(" ".join(sentences[start_idx:]))
    return raw_chunks


def apply_size_constraints(pieces, min_chunk_size=MIN_CHUNK_SIZE,
                            max_chunk_size=MAX_CHUNK_SIZE, hard_boundaries=None):
    hard_boundaries = hard_boundaries or set()

    merged = []
    buffer_chunk = ""
    for idx, c in enumerate(pieces):
        if idx in hard_boundaries and buffer_chunk:
            merged.append(buffer_chunk)
            buffer_chunk = ""
        buffer_chunk = (buffer_chunk + " " + c).strip() if buffer_chunk else c
        if len(buffer_chunk) >= min_chunk_size:
            merged.append(buffer_chunk)
            buffer_chunk = ""
    if buffer_chunk:
        if merged:
            merged[-1] += " " + buffer_chunk
        else:
            merged.append(buffer_chunk)

    final_chunks = []
    for c in merged:
        if len(c) > max_chunk_size:
            final_chunks.extend(split_oversized_chunk(c, max_chunk_size, FALLBACK_OVERLAP))
        else:
            final_chunks.append(c)
    return final_chunks


def hybrid_chunk_text(text, embed_fn,
                       min_chunk_size=MIN_CHUNK_SIZE,
                       max_chunk_size=MAX_CHUNK_SIZE,
                       structural_semantic_threshold=STRUCTURAL_UNIT_SEMANTIC_THRESHOLD):
    if not text.strip():
        return []

    units = split_structural_units(text)
    if not units:
        return []

    pieces = []
    hard_boundaries = set()
    for unit in units:
        if unit["is_heading_start"]:
            hard_boundaries.add(len(pieces))

        if len(unit["text"]) > structural_semantic_threshold:
            sub_pieces = semantic_split_unit(unit["text"], embed_fn)
        else:
            sub_pieces = [unit["text"]]

        pieces.extend(sub_pieces)

    return apply_size_constraints(
        pieces, min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size, hard_boundaries=hard_boundaries,
    )


def chunk_pages_semantically(pages, embed_fn):
    chunks = []
    global_index = 0
    for page in pages:
        page_chunks = hybrid_chunk_text(page["text"], embed_fn)
        for i, text in enumerate(page_chunks):
            chunks.append({
                "text": text,
                "page_number": page["page_number"],
                "chunk_id": f"p{page['page_number']}_c{i}_{global_index}",
                "chunk_index_in_page": i,
            })
            global_index += 1
    return chunks


# ---------------------------------------------------------
# 5. Store in Chroma DB and run a RAG query
# ---------------------------------------------------------
def main():
    print("Step 1: Setting up Chroma DB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    existing_count = collection.count()
    if existing_count > 0:
        print(f"   Collection already has {existing_count} chunks — skipping PDF reading, chunking, and embedding.\n")
    else:
        print("   Collection is empty.\n")

        print(f"Step 2: Reading '{PDF_PATH}'...")
        pages = extract_pdf_text(PDF_PATH)
        print(f"   Extracted text from {len(pages)} pages.\n")

        print("Step 3: Splitting text into hybrid (structural + semantic) chunks...")
        chunks = chunk_pages_semantically(pages, embed_fn=get_embedding)
        print(f"   Created {len(chunks)} chunks.\n")

        print("Step 4: Embedding and storing chunks...")
        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk["text"])
            collection.add(
                embeddings=[vector],
                documents=[chunk["text"]],
                metadatas=[{
                    "source": PDF_PATH,
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index_in_page": chunk["chunk_index_in_page"],
                }],
                ids=[f"{COLLECTION_NAME}_chunk_{i}"],
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                print(f"   [{i + 1}/{len(chunks)}] chunks processed.")
        print(f"\nStored {collection.count()} chunks in the '{COLLECTION_NAME}' collection.\n")

    # -----------------------------------------------------
    # 6. Interactive RAG query loop
    # -----------------------------------------------------
    print(f"Ready! Ask questions (type 'exit' or 'quit' to stop).\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break


        # Generate a hypothetical answer passage and embed THAT instead
        # of the raw question — closer in style/vocabulary to real chunks.
        hypothetical_doc = generate_hypothetical_document(user_question)
        retrieval_text = hypothetical_doc


        query_vector = get_embedding(retrieval_text)
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=N_RESULTS,
            include=["documents", "metadatas"],
        )

        matched_texts = search_results["documents"][0]
        matched_pages = [m["page_number"] for m in search_results["metadatas"][0]]

        context = "\n\n".join(
            f"(Page {p}): {t}" for t, p in zip(matched_texts, matched_pages)
        )
        

        # NOTE: the final answer is always grounded in the REAL retrieved
        # chunks and answers the user's ORIGINAL question — HyDE only
        # influenced which chunks got retrieved, never what gets said.
        system_prompt = f"""You are an assistant that analyzes academic papers.
Answer the user's question using ONLY the source excerpts provided below.
Synthesize across multiple excerpts if they're relevant — don't just quote
a single line if more context is available and useful.
Indicate which page(s) the information came from.
If the answer is not in the sources, say "This information is not found in the provided excerpts." Do not make anything up.

SOURCE EXCERPTS:
{context}"""

        answer = ask_llm(system_prompt=system_prompt, user_prompt=user_question)
        print(context)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Sources: page(s) {sorted(set(matched_pages))})\n")


if __name__ == "__main__":
    main()