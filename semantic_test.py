"""
Script that reads a PDF, splits it into chunks, embeds each chunk using
LM Studio, stores everything in a Chroma DB, and runs a RAG-based
question-answering query on top of it.

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
SEMANTIC_BUFFER_SIZE = 1            # how many neighboring sentences to fold in before embedding
SEMANTIC_BREAKPOINT_PERCENTILE = 90  # higher = fewer/bigger chunks, lower = more/smaller chunks
MIN_CHUNK_SIZE = 200                
MAX_CHUNK_SIZE = 1800               
FALLBACK_OVERLAP = 100


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
# 4. Semantic chunking
#    Splits text into sentences, embeds each sentence (with a small
#    neighbor buffer for stability), and cuts wherever the similarity
#    between consecutive sentences drops sharply (a topic shift).
# ---------------------------------------------------------
def split_sentences(text):
    # Basic sentence splitter. Good enough for academic prose; swap for
    # nltk/spacy if you need to handle abbreviations, citations, etc. better.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1 - sim


def split_oversized_chunk(text, max_size, overlap):
    """Fallback: hard-split any chunk that's still too big after semantic chunking."""
    pieces = []
    i = 0
    while i < len(text):
        piece = text[i : i + max_size].strip()
        if piece:
            pieces.append(piece)
        i += max_size - overlap
    return pieces


def semantic_chunk_text(text, embed_fn, buffer_size=SEMANTIC_BUFFER_SIZE,
                         breakpoint_percentile=SEMANTIC_BREAKPOINT_PERCENTILE,
                         min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    # Combine each sentence with its neighbors for a more stable embedding
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

    # Build chunks from the sentence breakpoints
    raw_chunks = []
    start_idx = 0
    for bp in breakpoints:
        raw_chunks.append(" ".join(sentences[start_idx : bp + 1]))
        start_idx = bp + 1
    raw_chunks.append(" ".join(sentences[start_idx:]))

    # Merge chunks that are too small
    merged = []
    buffer_chunk = ""
    for c in raw_chunks:
        buffer_chunk = (buffer_chunk + " " + c).strip() if buffer_chunk else c
        if len(buffer_chunk) >= min_chunk_size:
            merged.append(buffer_chunk)
            buffer_chunk = ""
    if buffer_chunk:
        if merged:
            merged[-1] += " " + buffer_chunk
        else:
            merged.append(buffer_chunk)

    # Split any chunk that's still too large
    final_chunks = []
    for c in merged:
        if len(c) > max_chunk_size:
            final_chunks.extend(split_oversized_chunk(c, max_chunk_size, FALLBACK_OVERLAP))
        else:
            final_chunks.append(c)

    return final_chunks


def chunk_pages_semantically(pages, embed_fn):
    """Runs semantic chunking per page (chunks never cross page boundaries),
    tagging each resulting chunk with its source page number."""
    chunks = []
    global_index = 0
    for page in pages:
        page_chunks = semantic_chunk_text(page["text"], embed_fn)
        for i, text in enumerate(page_chunks):
            chunks.append({
                "text": text,
                "page_number": page["page_number"],
                "chunk_id": f"p{page['page_number']}_c{i}_{global_index}",
                "chunk_index_in_page": i,
                "chunk_length": len(text),
            })
            global_index += 1
    return chunks
        

"""for text in page_chunks:
            chunks.append({"text": text, "page_number": page["page_number"]})
    return chunks"""
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

        print("Step 3: Splitting text into semantic chunks...")
        chunks = chunk_pages_semantically(pages, embed_fn=get_embedding)
        print(f"   Created {len(chunks)} semantic chunks.\n")

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
                    "chunk_length": chunk["chunk_length"],
                }],
                ids=[f"{COLLECTION_NAME}_chunk_{i}"],
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                print(f"   [{i + 1}/{len(chunks)}] chunks processed.")
        print(f"\nStored {collection.count()} chunks in the '{COLLECTION_NAME}' collection.\n")

    # -----------------------------------------------------
    # 6. Interactive RAG query loop — ask as many questions as you want
    # -----------------------------------------------------
    print("Ready! Ask questions about the paper (type 'exit' or 'quit' to stop).\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break

        query_vector = get_embedding(user_question)
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=4,
            include=["documents", "metadatas"],
        )

        matched_texts = search_results["documents"][0]
        matched_pages = [m["page_number"] for m in search_results["metadatas"][0]]

        context = "\n\n".join(
            f"(Page {p}): {t}" for t, p in zip(matched_texts, matched_pages)
        )

        system_prompt = f"""You are an assistant that analyzes academic papers.
Answer the user's question using ONLY the source excerpts provided below.
Indicate which page(s) the information came from.
If the answer is not in the sources, say "This information is not found in the provided excerpts." Do not make anything up.

SOURCE EXCERPTS:
{context}"""

        answer = ask_llm(system_prompt=system_prompt, user_prompt=user_question)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Sources: page(s) {sorted(set(matched_pages))})\n")


if __name__ == "__main__":
    main()