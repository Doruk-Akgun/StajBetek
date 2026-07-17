
import json
import os
import re
import time
import chromadb
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
PDF_PATH = "16.pdf"
CHROMA_DB_PATH = "./general_experiment"
COLLECTION_NAME = "academic_paper_2"

PARENT_CHUNK_SIZE = 2000
PARENT_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 60

VECTOR_TOP_K = 15
BM25_TOP_K = 15
FINAL_TOP_K = 5
RRF_K = 60  # standard RRF damping constant


SEARCH_MODE = "hybrid"

VALID_MODES = (
    "semantic",
    "semantic_parent_child",
    "hybrid",
    "hybrid_parent_child",
)


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
# 4. Generic character-based splitter (used for both parent and child chunks)
# ---------------------------------------------------------
def split_into_pieces(text, chunk_size, overlap):
    pieces = []
    i = 0
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        if piece:
            pieces.append(piece)
        i += chunk_size - overlap
    return pieces


# ---------------------------------------------------------
# 4b. Parent-child chunking
# ---------------------------------------------------------
def chunk_text_parent_child(pages):
    parent_chunks = []
    child_chunks = []

    parent_counter = 0
    for page in pages:
        parent_pieces = split_into_pieces(page["text"], PARENT_CHUNK_SIZE, PARENT_OVERLAP)
        for parent_text in parent_pieces:
            parent_id = f"parent_{parent_counter}"
            parent_counter += 1
            parent_chunks.append(
                {"parent_id": parent_id, "text": parent_text, "page_number": page["page_number"]}
            )

            child_pieces = split_into_pieces(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
            for i, text in enumerate(child_pieces):
                child_chunks.append(
                    {"parent_id": parent_id, "text": text,
                    "page_number": page["page_number"],
                    "chunk_index_in_page": i, 
                    "chunk_length": len(text)}
                )

    return parent_chunks, child_chunks


# ---------------------------------------------------------
# 4c. Tokenizer for BM25 (simple, dependency-free)aaaa
# ---------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------
# 4d. Build a BM25 index over every child chunk currently in Chroma
# ---------------------------------------------------------
def build_bm25_index(collection):
    all_children = collection.get(include=["documents", "metadatas"])
    ids = all_children["ids"]
    documents = all_children["documents"]
    metadatas = all_children["metadatas"]

    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


# ---------------------------------------------------------
# 4e. Reciprocal Rank Fusion
# ---------------------------------------------------------
def reciprocal_rank_fusion(ranked_id_lists, k=RRF_K):
    scores = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in fused]


# ---------------------------------------------------------
# 4f. Retrieval mode: semantic-only (dense vector) search.
#     Returns a ranked list of {"id", "text", "metadata"} records.
# ---------------------------------------------------------
def semantic_search(user_question, collection, final_n=FINAL_TOP_K):
    query_vector = get_embedding(user_question)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=final_n,
        include=["documents", "metadatas"],
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [{"id": i, "text": d, "metadata": m} for i, d, m in zip(ids, documents, metadatas)]


# ---------------------------------------------------------
# 4g. Retrieval mode: hybrid (vector + BM25, fused with RRF).
#     Returns a ranked list of {"id", "text", "metadata"} records.
# ---------------------------------------------------------
def hybrid_search(user_question, collection, bm25_index, final_n=FINAL_TOP_K):
    # --- Vector (dense) search ---
    query_vector = get_embedding(user_question)
    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=VECTOR_TOP_K,
        include=["metadatas"],
    )
    vector_ranked_ids = vector_results["ids"][0]

    # --- BM25 (sparse/keyword) search ---
    tokenized_query = tokenize(user_question)
    bm25_scores = bm25_index["bm25"].get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:BM25_TOP_K]
    bm25_ranked_ids = [bm25_index["ids"][i] for i in ranked_indices]

    # --- Fuse the two rankings ---
    fused_ids = reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids])
    top_ids = fused_ids[:final_n]

    # Resolve text + metadata for the fused top ids via a lookup
    # (don't rely on Chroma's .get() id ordering).
    fetched = collection.get(ids=top_ids, include=["documents", "metadatas"])
    id_to_record = {
        doc_id: {"id": doc_id, "text": doc, "metadata": meta}
        for doc_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }

    return [id_to_record[doc_id] for doc_id in top_ids if doc_id in id_to_record]


# ---------------------------------------------------------
# 4h. Context builder A: "flat" — use the matched CHILD chunk text
#     directly as context (no parent expansion).
# ---------------------------------------------------------
def build_context(records):
    pages = [r["metadata"]["page_number"] for r in records]
    context = "\n\n".join(f"(Page {r['metadata']['page_number']}): {r['text']}" for r in records)
    return context, pages


# ---------------------------------------------------------
# 4i. Context builder B: "parent_child" — expand matched child chunks
#     up to their PARENT chunk text as context.
# ---------------------------------------------------------
def build_context_parent_child(records, parent_collection):
    matched_parent_ids = [r["metadata"]["parent_id"] for r in records]
    unique_parent_ids = list(dict.fromkeys(matched_parent_ids))

    parents = parent_collection.get(ids=unique_parent_ids, include=["documents", "metadatas"])
    parent_texts = parents["documents"]
    parent_pages = [m["page_number"] for m in parents["metadatas"]]

    context = "\n\n".join(f"(Page {p}): {t}" for t, p in zip(parent_texts, parent_pages))
    return context, parent_pages


# ---------------------------------------------------------
# 4j. Mode dispatch (switch/case via match statement).
#     Runs the right retriever + the right context builder for a mode
#     and returns (context, pages, records, elapsed_seconds).
# ---------------------------------------------------------
def run_mode(mode, user_question, collection, parent_collection, bm25_index):
    start = time.perf_counter()

    match mode:
        case "semantic":
            records = semantic_search(user_question, collection, final_n=FINAL_TOP_K)
            context, pages = build_context(records)

        case "semantic_parent_child":
            records = semantic_search(user_question, collection, final_n=FINAL_TOP_K)
            context, pages = build_context_parent_child(records, parent_collection)

        case "hybrid":
            records = hybrid_search(user_question, collection, bm25_index, final_n=FINAL_TOP_K)
            context, pages = build_context(records)

        case "hybrid_parent_child":
            records = hybrid_search(user_question, collection, bm25_index, final_n=FINAL_TOP_K)
            context, pages = build_context_parent_child(records, parent_collection)

        case _:
            raise ValueError(f"Unknown SEARCH_MODE: {mode!r}. Valid modes: {VALID_MODES}")

    elapsed = time.perf_counter() - start
    return context, pages, records, elapsed


def answer_with_context(user_question, context):
    system_prompt = f"""You are an assistant that answers questions about academic papers.

Use ONLY the provided source excerpts.

Rules:
1. Never use outside knowledge.
2. If the answer is not in the excerpts, say:
   "This information is not found in the provided excerpts."
3. Always cite the page number(s).
4. When the excerpts contain a table:
   - Read rows and columns carefully.
   - Do not combine values from different rows or different tables.
   - If answering from a table, identify the matching row before answering.
5. If multiple excerpts contain similar tables, use the table that directly answers the question.
6. Think carefully before answering, but output only the final answer.
"""
    user_prompt = f"""
Question:
{user_question}

SOURCE EXCERPTS:
{context}
"""

    return ask_llm(system_prompt=system_prompt, user_prompt=user_prompt)


# ---------------------------------------------------------
# 5. Store in Chroma DB and run the RAG query loop
# ---------------------------------------------------------
def main():
    print("Step 1: Setting up Chroma DB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    parent_collection = chroma_client.get_or_create_collection(name=f"{COLLECTION_NAME}_parents")

    existing_count = collection.count()
    if existing_count > 0:
        print(f"   Collection already has {existing_count} child chunks — skipping PDF reading, chunking, and embedding.\n")
    else:
        print("   Collection is empty.\n")

        print(f"Step 2: Reading '{PDF_PATH}'...")
        pages = extract_pdf_text(PDF_PATH)
        print(f"   Extracted text from {len(pages)} pages.\n")

        print("Step 3: Splitting text into parent and child chunks...")
        parent_chunks, child_chunks = chunk_text_parent_child(pages)
        print(f"   Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks.\n")

        print("Step 4a: Storing parent chunks (text only, no embedding needed)...")
        for parent in parent_chunks:
            parent_collection.add(
                embeddings=[[0.0]],
                documents=[parent["text"]],
                metadatas=[{"source": PDF_PATH, "page_number": parent["page_number"]}],
                ids=[parent["parent_id"]],
            )
        print(f"   Stored {parent_collection.count()} parent chunks.\n")

        print("Step 4b: Embedding and storing child chunks...")
        for i, child in enumerate(child_chunks):
            vector = get_embedding(child["text"])
            collection.add(
                embeddings=[vector],
                documents=[child["text"]],
                metadatas=[{
                    "source": PDF_PATH,
                    "page_number": child["page_number"],
                    "parent_id": child["parent_id"],
                    "chunk_index_in_page": child["chunk_index_in_page"], 
                    "chunk_length": child["chunk_length"]
                }],
                ids=[f"{COLLECTION_NAME}_child_{i}"],
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(child_chunks):
                print(f"   [{i + 1}/{len(child_chunks)}] child chunks processed.")
        print(f"\nStored {collection.count()} child chunks in the '{COLLECTION_NAME}' collection.\n")

    print("Step 5: Building BM25 keyword index over child chunks...")
    bm25_index = build_bm25_index(collection)
    print(f"   BM25 index built over {len(bm25_index['ids'])} child chunks.\n")

    print(f"Active SEARCH_MODE: {SEARCH_MODE}")
    print("Ready! Ask questions about the paper (type 'exit'/'quit' to stop).")
    print("Prefix a question with '/compare ' to run ALL 4 modes on it and compare.\n")

    while True:
        raw_input_text = input("Your question: ").strip()
        if raw_input_text.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break

        if raw_input_text.lower().startswith("/compare "):
            user_question = raw_input_text[len("/compare "):].strip()
            compare_all_modes(user_question, collection, parent_collection, bm25_index)
            continue

        user_question = raw_input_text
        context, pages, _records, elapsed = run_mode(
            SEARCH_MODE, user_question, collection, parent_collection, bm25_index
        )

        print(context)
        answer = answer_with_context(user_question, context)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Mode: {SEARCH_MODE} | retrieval time: {elapsed:.3f}s | Sources: page(s) {sorted(set(pages))})\n")


# ---------------------------------------------------------
# 6. Compare all 4 modes side by side on the same question
# ---------------------------------------------------------
def compare_all_modes(user_question, collection, parent_collection, bm25_index):
    print(f"\n=== Comparing all 4 modes for: {user_question!r} ===\n")

    results = {}
    for mode in VALID_MODES:
        context, pages, records, elapsed = run_mode(
            mode, user_question, collection, parent_collection, bm25_index
        )
        answer = answer_with_context(user_question, context)
        results[mode] = {
            "context": context,
            "pages": sorted(set(pages)),
            "num_chunks": len(records),
            "elapsed": elapsed,
            "answer": answer,
        }

    for mode in VALID_MODES:
        r = results[mode]
        print(f"--- {mode} ---")
        print(f"retrieval time: {r['elapsed']:.3f}s | chunks used: {r['num_chunks']} | pages: {r['pages']}")
        print(f"answer: {r['answer']}")
        print()

    print("=== End comparison ===\n")


if __name__ == "__main__":
    main()