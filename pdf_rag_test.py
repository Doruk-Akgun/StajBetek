"""
Script that reads a PDF, splits it into chunks, embeds each chunk using
LM Studio, stores everything in a Chroma DB, and runs a RAG-based
question-answering query on top of it.

Requirements:
    pip install pypdf chromadb requests
"""

import json
import os
import chromadb
import requests
from pypdf import PdfReader


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"   # no trailing slash
PDF_PATH = "16.pdf"                                # path to your PDF file
CHROMA_DB_PATH = "./experimental_rag_db"           # same DB, dedicated collection
COLLECTION_NAME = "academic_paper_16"              # collection specific to this paper
CHUNK_SIZE = 900            # in characters
CHUNK_OVERLAP = 150         # overlap between consecutive chunks
USER_QUESTION = "When can be done in the future about heart disease detection?"


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
# 4. Split text into overlapping chunks (character-based)
# ---------------------------------------------------------
def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for page in pages:
        text = page["text"]
        i = 0
        while i < len(text):
            piece = text[i : i + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "page_number": page["page_number"]})
            i += chunk_size - overlap
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
 
        print("Step 3: Splitting text into chunks...")
        chunks = chunk_text(pages)
        print(f"   Created {len(chunks)} chunks.\n")
 
        print("Step 4: Embedding and storing chunks...")
        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk["text"])
            collection.add(
                embeddings=[vector],
                documents=[chunk["text"]],
                metadatas=[{"source": PDF_PATH, "page_number": chunk["page_number"]}],
                ids=[f"{COLLECTION_NAME}_chunk_{i}"],
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                print(f"   [{i + 1}/{len(chunks)}] chunks processed.")
        print(f"\nStored {collection.count()} chunks in the '{COLLECTION_NAME}' collection.\n")

    # -----------------------------------------------------
    # 6. RAG query — example question
    # -----------------------------------------------------
    user_question = USER_QUESTION
    print(f"User Question: {user_question}\n")

    query_vector = get_embedding(user_question)
    search_results = collection.query(
        query_embeddings=[query_vector],
        n_results=4,
        include=["documents", "metadatas"],
    )

    retrieved_chunks = search_results["documents"][0]
    source_pages = [m["page_number"] for m in search_results["metadatas"][0]]

    print("Most relevant retrieved chunks (with page numbers):")
    for chunk, page in zip(retrieved_chunks, source_pages):
        print(f"  [Page {page}] {chunk[:100]}...")
    print()

    context = "\n\n".join(
        f"(Page {p}): {c}" for c, p in zip(retrieved_chunks, source_pages)
    )

    system_prompt = f"""You are an assistant that analyzes academic papers.
Answer the user's question using ONLY the source excerpts provided below.
Indicate which page(s) the information came from.
If the answer is not in the sources, say "This information is not found in the provided excerpts." Do not make anything up.

SOURCE EXCERPTS:
{context}"""

    answer = ask_llm(system_prompt=system_prompt, user_prompt=user_question)

    print("=== RAG SYSTEM ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
