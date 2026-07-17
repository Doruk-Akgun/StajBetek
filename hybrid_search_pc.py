"""
Script that reads a PDF, splits it into chunks, embeds each chunk using
LM Studio, stores everything in a Chroma DB, and runs a HYBRID
(vector + BM25 keyword) RAG-based question-answering query on top of it.

Hybrid search = dense vector search (semantic) + BM25 sparse search
(exact keyword / acronym / number matching), merged with Reciprocal
Rank Fusion (RRF). This catches both "what does this mean" style
matches and "find this exact term" style matches that embeddings
alone often miss.

Requires: pip install rank_bm25 --break-system-packages
"""

import json
import os
import re
import chromadb
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
PDF_PATH = "16.pdf"
CHROMA_DB_PATH = "./experimental_hybrid_search_db"
COLLECTION_NAME = "academic_paper_1"

PARENT_CHUNK_SIZE = 3000
PARENT_OVERLAP = 400
CHILD_CHUNK_SIZE = 800
CHILD_OVERLAP = 100

# How many candidates each retriever contributes before fusion,
# and how many fused results get sent to the LLM as context.
VECTOR_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 5
RRF_K = 60  # standard RRF damping constant


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
            for child_text in child_pieces:
                child_chunks.append(
                    {"parent_id": parent_id, "text": child_text, "page_number": page["page_number"]}
                )

    return parent_chunks, child_chunks


# ---------------------------------------------------------
# 4c. Tokenizer for BM25 (simple, dependency-free)
# ---------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------
# 4d. Build a BM25 index over every child chunk currently in Chroma
#     Rebuilt at startup each run — cheap for a single-paper corpus.
#     For large corpora, persist the tokenized corpus instead of
#     re-fetching/re-tokenizing every time.
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
#     Combines multiple ranked id lists into one fused ranking.
#     Works well even though vector distances and BM25 scores
#     live on completely different scales.
# ---------------------------------------------------------
def reciprocal_rank_fusion(ranked_id_lists, k=RRF_K):
    scores = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in fused]


# ---------------------------------------------------------
# 4f. Hybrid retrieval: vector search + BM25, fused with RRF
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

    # Resolve metadata (parent_id, page_number) for the fused top ids.
    # Build a lookup instead of relying on Chroma's .get() id ordering.
    id_to_metadata = {}
    fetched = collection.get(ids=top_ids, include=["metadatas"])
    for doc_id, meta in zip(fetched["ids"], fetched["metadatas"]):
        id_to_metadata[doc_id] = meta

    child_metadatas = [id_to_metadata[doc_id] for doc_id in top_ids if doc_id in id_to_metadata]
    return child_metadatas


# ---------------------------------------------------------
# 5. Store in Chroma DB and run a hybrid RAG query
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
                }],
                ids=[f"{COLLECTION_NAME}_child_{i}"],
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(child_chunks):
                print(f"   [{i + 1}/{len(child_chunks)}] child chunks processed.")
        print(f"\nStored {collection.count()} child chunks in the '{COLLECTION_NAME}' collection.\n")

    print("Step 5: Building BM25 keyword index over child chunks...")
    bm25_index = build_bm25_index(collection)
    print(f"   BM25 index built over {len(bm25_index['ids'])} child chunks.\n")

    # -----------------------------------------------------
    # 6. Interactive hybrid RAG query loop
    # -----------------------------------------------------
    print("Ready! Ask questions about the paper (type 'exit' or 'quit' to stop).\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break

        matched_child_metadatas = hybrid_search(user_question, collection, bm25_index)

        matched_parent_ids = [m["parent_id"] for m in matched_child_metadatas]
        unique_parent_ids = list(dict.fromkeys(matched_parent_ids))

        parents = parent_collection.get(ids=unique_parent_ids, include=["documents", "metadatas"])
        parent_texts = parents["documents"]
        parent_pages = [m["page_number"] for m in parents["metadatas"]]

        context = "\n\n".join(
            f"(Page {p}): {t}" for t, p in zip(parent_texts, parent_pages)
        )

        system_prompt = f"""You are an assistant that analyzes academic papers.
Answer the user's question using ONLY the source excerpts provided below.
Indicate which page(s) the information came from.
If the answer is not in the sources, say "This information is not found in the provided excerpts." Do not make anything up.

SOURCE EXCERPTS:
{context}"""

        print(context)
        answer = ask_llm(system_prompt=system_prompt, user_prompt=user_question)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Sources: page(s) {sorted(set(parent_pages))})\n")


if __name__ == "__main__":
    main()