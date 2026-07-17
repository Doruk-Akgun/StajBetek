

import json
import os
import chromadb
import requests
from pypdf import PdfReader


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
PDF_PATH = "16.pdf"
CHROMA_DB_PATH = "./experimental_rag_db"
COLLECTION_NAME = "academic_paper_18"

PARENT_CHUNK_SIZE = 3000
PARENT_OVERLAP = 400
CHILD_CHUNK_SIZE = 800
CHILD_OVERLAP = 100


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
#     Each parent chunk is split further into smaller child chunks.
#     Children are what gets embedded/searched; parents are what gets
#     sent to the LLM as context.
# ---------------------------------------------------------
def chunk_text_parent_child(pages):
    parent_chunks = []   #METADATA {"parent_id", "text", "page_number"}
    child_chunks = []    #METADATA {"parent_id", "text", "page_number"}

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
# 5. Store in Chroma DB and run a RAG query
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
                embeddings=[[0.0]],  # dummy vector — this collection is never searched, only fetched by id
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
            include=["metadatas"],
        )

        # Resolve the unique parent chunks behind the matched children,
        # preserving the order they were matched in.
        matched_parent_ids = [m["parent_id"] for m in search_results["metadatas"][0]]
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

        answer = ask_llm(system_prompt=system_prompt, user_prompt=user_question)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Sources: page(s) {sorted(set(parent_pages))})\n")


if __name__ == "__main__":
    main()