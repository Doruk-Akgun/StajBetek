"""
Same parent-child RAG pipeline as before, but using Qdrant instead of
Chroma as the vector database.

Requirements:
    pip install pypdf qdrant-client requests
"""

import json
import os
import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
PDF_PATH = "16.pdf"

# Embedded (local file) mode — closest equivalent to Chroma's PersistentClient.
# Only one process can open this path at a time.
# If you're running a Qdrant server instead (e.g. via Docker), replace this
# with: QDRANT_CLIENT_KWARGS = {"url": "http://localhost:6333"}
QDRANT_PATH = "./experimental_rag_qdrant_db"

CHILD_COLLECTION = "academic_paper_16_children"
PARENT_COLLECTION = "academic_paper_16_parents"

PARENT_CHUNK_SIZE = 2000
PARENT_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 50


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
# 4. Generic character-based splitter
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
    parent_chunks = []   # {"parent_id", "text", "page_number"}
    child_chunks = []    # {"parent_id", "text", "page_number"}

    parent_id = 0
    for page in pages:
        parent_pieces = split_into_pieces(page["text"], PARENT_CHUNK_SIZE, PARENT_OVERLAP)
        for parent_text in parent_pieces:
            parent_chunks.append(
                {"parent_id": parent_id, "text": parent_text, "page_number": page["page_number"]}
            )

            child_pieces = split_into_pieces(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
            for child_text in child_pieces:
                child_chunks.append(
                    {"parent_id": parent_id, "text": child_text, "page_number": page["page_number"]}
                )

            parent_id += 1  # Qdrant point ids must be int or UUID — plain ints work fine here

    return parent_chunks, child_chunks


# ---------------------------------------------------------
# 5. Store in Qdrant and run a RAG query
# ---------------------------------------------------------
def main():
    print("Step 1: Setting up Qdrant client...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    # Qdrant needs to know the vector size upfront, so we get it from one
    # real embedding call before creating the collection.


    if not qdrant_client.collection_exists(CHILD_COLLECTION):
        probe_vector = get_embedding("dimension probe")
        embedding_dim = len(probe_vector)
        qdrant_client.create_collection(
            collection_name=CHILD_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    if not qdrant_client.collection_exists(PARENT_COLLECTION):
        # Parents are never searched by vector, only fetched by id — a
        # 1-dim dummy vector config is enough.
        qdrant_client.create_collection(
            collection_name=PARENT_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )

    existing_count = qdrant_client.count(collection_name=CHILD_COLLECTION, exact=True).count
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

        print("Step 4a: Storing parent chunks (text only, no real embedding needed)...")
        parent_points = [
            PointStruct(
                id=parent["parent_id"],
                vector=[0.0],
                payload={"text": parent["text"], "source": PDF_PATH, "page_number": parent["page_number"]},
            )
            for parent in parent_chunks
        ]
        qdrant_client.upsert(collection_name=PARENT_COLLECTION, points=parent_points)
        print(f"   Stored {qdrant_client.count(collection_name=PARENT_COLLECTION).count} parent chunks.\n")

        print("Step 4b: Embedding and storing child chunks...")
        for i, child in enumerate(child_chunks):
            vector = get_embedding(child["text"])
            qdrant_client.upsert(
                collection_name=CHILD_COLLECTION,
                points=[
                    PointStruct(
                        id=i,
                        vector=vector,
                        payload={
                            "text": child["text"],
                            "source": PDF_PATH,
                            "page_number": child["page_number"],
                            "parent_id": child["parent_id"],
                        },
                    )
                ],
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(child_chunks):
                print(f"   [{i + 1}/{len(child_chunks)}] child chunks processed.")
        print(f"\nStored {qdrant_client.count(collection_name=CHILD_COLLECTION).count} child chunks in '{CHILD_COLLECTION}'.\n")

    # -----------------------------------------------------
    # 6. Interactive RAG query loop
    # -----------------------------------------------------
    print("Ready! Ask questions about the paper (type 'exit' or 'quit' to stop).\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break

        query_vector = get_embedding(user_question)
        search_results = qdrant_client.query_points(
            collection_name=CHILD_COLLECTION,
            query=query_vector,
            limit=4,
            with_payload=True,
        ).points

        matched_parent_ids = [hit.payload["parent_id"] for hit in search_results]
        unique_parent_ids = list(dict.fromkeys(matched_parent_ids))

        parents = qdrant_client.retrieve(
            collection_name=PARENT_COLLECTION,
            ids=unique_parent_ids,
            with_payload=True,
        )
        parent_texts = [p.payload["text"] for p in parents]
        parent_pages = [p.payload["page_number"] for p in parents]

        context = "\n\n".join(
            f"(Page {p}): {t}" for t, p in zip(parent_texts, parent_pages)
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
        print(f"(Sources: page(s) {sorted(set(parent_pages))})\n")


if __name__ == "__main__":
    main()