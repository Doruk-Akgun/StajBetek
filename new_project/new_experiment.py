import json
import os
import re
import time
import chromadb
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from pdf_reading_order import extract_pdf_text


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

DATASET_PATH = "dataset"
PDF_FILES =["TDS_AQUALUX.pdf",
            "TDS_EXXENMAT.pdf",
            "TDS_MOMENTOPLASTIX.pdf",
            "TDS_WOODMAXX_WOODSTAIN.pdf",
            "TDS_MOMENTOSILAN.pdf"]

PDF_PATHS = [os.path.join(DATASET_PATH, file) for file in PDF_FILES]


CHROMA_DB_PATH = "./paint_db"
COLLECTION_NAME = "test1"

PARENT_CHUNK_SIZE = 1024
PARENT_OVERLAP = 128
CHILD_CHUNK_SIZE = 256
CHILD_OVERLAP = 32

VECTOR_TOP_K = 15
BM25_TOP_K = 15
FINAL_TOP_K = 5
RRF_K = 60  # standard RRF damping constant


SEARCH_MODE = "hybrid_parent_child"

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
def chunk_text_parent_child(pages, doc_id):
    parent_chunks = []
    child_chunks = []

    parent_counter = 0
    for page in pages:
        parent_pieces = split_into_pieces(page["text"], PARENT_CHUNK_SIZE, PARENT_OVERLAP)
        for parent_text in parent_pieces:
            parent_id = f"{doc_id}_parent_{parent_counter}"
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
# 4b-2. Ingest a single PDF into an existing (possibly non-empty) collection.
#     Safe to call multiple times with different files: parent/child ids
#     are namespaced with doc_id so they won't collide with earlier files.
# ---------------------------------------------------------
def ingest_pdf(pdf_path, collection, parent_collection, pdf_file):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"   Reading '{pdf_path}'...")
    pages = extract_pdf_text(pdf_path)
    print(f"   Extracted text from {len(pages)} pages.")

    parent_chunks, child_chunks = chunk_text_parent_child(pages, doc_id)
    print(f"   Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks.")

    for parent in parent_chunks:
        parent_collection.add(
            embeddings=[[0.0]],
            documents=[parent["text"]],
            metadatas=[{"source": pdf_file, "page_number": parent["page_number"]}],
            ids=[parent["parent_id"]],
        )

    for i, child in enumerate(child_chunks):
        vector = get_embedding(child["text"])
        child_id = f"{doc_id}_child_{i}"
        collection.add(
            embeddings=[vector],
            documents=[child["text"]],
            metadatas=[{
                "source": pdf_file,
                "page_number": child["page_number"],
                "child_id": child_id,
                "parent_id": child["parent_id"],
                "chunk_index_in_page": child["chunk_index_in_page"],
                "chunk_length": child["chunk_length"],
            }],
            ids=[child_id],
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(child_chunks):
            print(f"   [{i + 1}/{len(child_chunks)}] child chunks processed.")

    print(f"   Finished ingesting '{pdf_path}': {len(parent_chunks)} parents, {len(child_chunks)} children.\n")


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
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [{"id": i, "text": d, "metadata": m, "distance": dist} for i, d, m, dist in zip(ids, documents, metadatas, distances)]


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
        include=["metadatas", "distances"],
    )
    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_ranked_ids = vector_ids  # no filtering, just ranking as before

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

    # Build lookup dicts so we can attach scores to the final records
    dense_score_map = dict(zip(vector_ids, vector_distances))
    sparse_score_map = {bm25_index["ids"][i]: bm25_scores[i] for i in ranked_indices}

    fetched = collection.get(ids=top_ids, include=["documents", "metadatas"])
    id_to_record = {
        doc_id: {
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "dense_distance": dense_score_map.get(doc_id),
            "sparse_score": sparse_score_map.get(doc_id),
        }
        for doc_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }

    return [id_to_record[doc_id] for doc_id in top_ids if doc_id in id_to_record]


# ---------------------------------------------------------
# 4h. Context builder A: "flat" — use the matched CHILD chunk text
#     directly as context (no parent expansion).
# ---------------------------------------------------------
def build_context(records):
    pages = [r["metadata"]["page_number"] for r in records]
    context = "\n\n".join(f"Source: {r['metadata']['source']}\n(Page {r['metadata']['page_number']}): {r['text']}" for r in records)
    for i, r in enumerate(records):
        print(f"\n===== Chunk {i+1} =====")
        print(f"Source: {r['metadata']['source']}")
        print(f"Page: {r['metadata']['page_number']}")
        if "distance" in r:
            print(f"Dense distance: {r['distance']:.4f}")
        if r.get("dense_distance") is not None:
            print(f"Dense distance: {r['dense_distance']:.4f}")
        if r.get("sparse_score") is not None:
            print(f"Sparse score: {r['sparse_score']:.4f}")
        print(r["text"])
    return context, pages


# ---------------------------------------------------------
# 4i. Context builder B: "parent_child" — expand matched child chunks
#     up to their PARENT chunk text as context.
# ---------------------------------------------------------
def build_context_parent_child(records, parent_collection):
    matched_parent_ids = [r["metadata"]["parent_id"] for r in records]
    unique_parent_ids = list(dict.fromkeys(matched_parent_ids))

    parents = parent_collection.get(ids=unique_parent_ids, include=["documents", "metadatas"])
    parent_pages = [m["page_number"] for m in parents["metadatas"]]

    print(unique_parent_ids)
    print(parents["ids"])

    parent_map = {
        pid: (doc, meta)
        for pid, doc, meta in zip(
            parents["ids"],
            parents["documents"],
            parents["metadatas"]
        )
    }

    contexts = []

    for record in records:
        pid = record["metadata"]["parent_id"]
        doc, meta = parent_map[pid]

        contexts.append(
            f"""Source: {meta['source']}
    Page: {meta['page_number']}

    {doc}
    """
        )

    context = "\n\n".join(contexts)
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
    system_prompt = f"""You are a document assistant.

Use ONLY the provided source excerpts.

The user's input may be:
- a question,
- a product name,
- a keyword,
- a specification,
- or a short phrase.

If the input is not a complete question, treat it as a request to summarize or explain the matching information from the excerpts or return names of matching documents.

If the excerpts contain the requested information:
- answer clearly,
- combine relevant excerpts when appropriate,
- cite the page numbers.

If the excerpts do not contain the requested information, reply exactly:

"This information is not found in the provided excerpts."

Never use knowledge outside the excerpts.
Never fabricate missing information.
"""
    user_prompt = f"""
User request:
{user_question}

Relevant source excerpts:
{context}

Respond according to the system instructions.
"""

    return ask_llm(system_prompt=system_prompt, user_prompt=user_prompt)


# ---------------------------------------------------------
# 5. Store in Chroma DB and run the RAG query loop
# ---------------------------------------------------------
def main():
    print("Step 1: Setting up Chroma DB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    parent_collection = chroma_client.get_or_create_collection(name=f"{COLLECTION_NAME}_parents")

    existing_count = collection.count()
    print(f"   Collection currently has {existing_count} child chunks.\n")

    print("Step 2: Checking which files still need to be ingested...")
    for pdf_file in PDF_FILES:
        pdf_path = os.path.join(DATASET_PATH, pdf_file)
        already_ingested = collection.get(where={"source": pdf_file}, limit=1)["ids"]
        if already_ingested:
            print(f"   '{pdf_path}' already ingested — skipping.")
        else:
            print(f"   '{pdf_path}' not found in collection — ingesting now...")
            ingest_pdf(pdf_path, collection, parent_collection, pdf_file)

    print(f"Collection now has {collection.count()} child chunks and {parent_collection.count()} parent chunks in '{COLLECTION_NAME}'.\n")

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
        context, pages, records, elapsed = run_mode(
            SEARCH_MODE, user_question, collection, parent_collection, bm25_index
        )

        print(context)
        answer = answer_with_context(user_question, context)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Mode: {SEARCH_MODE} | retrieval time: {elapsed:.3f}s | Sources: page(s) {sorted(set(pages))})\n")
        for r in records:
            print(r["metadata"]["parent_id"])

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