from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import json
import os
import re
import faiss
import numpy as np
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader, 
    TextLoader
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

genai.configure(api_key="AIzaSyACLTk3la54Go1JP84NnVlPwiUO-dkFpHI")
model_gen = genai.GenerativeModel("gemini-2.5-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

upload_dir = os.path.join(BASE_DIR, "uploads")
metadata_file = os.path.join(BASE_DIR, "metadata.json")
faiss_file = os.path.join(BASE_DIR, "faiss.index")
chunks_file = os.path.join(BASE_DIR, "chunk_metadata.json")

os.makedirs(upload_dir, exist_ok=True)
if not os.path.exists(chunks_file):
    with open(chunks_file, "w") as f:
        json.dump([], f)

if not os.path.exists(metadata_file):
    with open(metadata_file, "w") as f:
        json.dump({}, f)


def load_metadata():
    with open(metadata_file, "r") as f:
        return json.load(f)
    
def save_metadata(data):
    with open(metadata_file, "w") as f:
        json.dump(data, f, indent=4)

def load_chunks_metadata():
    with open(chunks_file, "r") as f:
        return json.load(f)
    
def save_chunks_metadata(data):
    with open(chunks_file, "w") as f:
        json.dump(data, f, indent=4)

def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text

def generate_embeddings(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return np.array(embeddings, dtype=np.float32)

def load_document(document_id: str):
    metadata = load_metadata()
    if document_id not in metadata:
        raise Exception("Document not found")
    
    file_path = metadata[document_id]["path"]
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise Exception(f"Unsupported file type: {ext}")

    docs = loader.load()
    return docs

def chunk_document(text):
    splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    text = clean_text(text)
    chunks = splitter.split_text(text)
    return chunks

def detect_document_type(text):
    text_lower = text.lower()
    
    invoice_keywords = ["invoice", "bill", "amount due", "vendor", "invoice number", "total", "subtotal", "tax"]
    medical_keywords = ["patient", "diagnosis", "medical report", "doctor", "hospital", "symptoms", "treatment", "vital signs", "dob", "date of birth"]
    resume_keywords = ["experience", "education", "skills", "employment", "work experience", "career", "qualifications", "professional"]
    
    invoice_count = sum(1 for kw in invoice_keywords if kw in text_lower)
    medical_count = sum(1 for kw in medical_keywords if kw in text_lower)
    resume_count = sum(1 for kw in resume_keywords if kw in text_lower)
    
    doc_type = max([("invoice", invoice_count), ("medical_report", medical_count), ("resume", resume_count)], key=lambda x: x[1])
    return doc_type[0] if doc_type[1] > 0 else "unknown"

def extract_document_info(text, doc_type):
    lines = text.split("\n")
    text_lower = text.lower()
    
    if doc_type == "invoice":
        info = {
            "type": "Invoice",
            "invoice_number": extract_pattern(text, r"invoice[#\s]*[:]*\s*([A-Z0-9\-]+)", "N/A"),
            "date": extract_pattern(text, r"date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "N/A"),
            "vendor": extract_first_after_keyword(lines, ["vendor", "from", "company"], "N/A"),
            "total_amount": extract_pattern(text, r"(?:total|amount due)[:\s]*\$*([\d,.]+)", "N/A"),
            "items_count": len([l for l in lines if any(c.isdigit() for c in l) and len(l) > 5])
        }
    elif doc_type == "medical_report":
        info = {
            "type": "Medical Report",
            "patient_name": extract_first_after_keyword(lines, ["patient", "name"], "N/A"),
            "date_of_birth": extract_pattern(text, r"(?:dob|date of birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "N/A"),
            "doctor": extract_first_after_keyword(lines, ["doctor", "physician", "provider"], "N/A"),
            "diagnosis": extract_first_after_keyword(lines, ["diagnosis", "condition"], "N/A"),
            "date": extract_pattern(text, r"(?:report date|date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "N/A")
        }
    elif doc_type == "resume":
        info = {
            "type": "Resume",
            "name": extract_first_non_empty_line(lines),
            "email": extract_pattern(text, r"[\w\.-]+@[\w\.-]+\.\w+", "N/A"),
            "phone": extract_pattern(text, r"(\d{10}|\d{3}[.-]\d{3}[.-]\d{4}|\+\d{1,3}[.\s]?\d{1,14})", "N/A"),
            "experience_years": count_keyword_occurrences(text_lower, ["year", "years"]),
            "skills_mentioned": count_unique_skills(text)
        }
    else:
        info = {"type": "Unknown Document", "content_length": len(text)}
    
    return info

def extract_pattern(text, pattern, default):
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return default
    if match.groups():
        return match.group(1).strip()
    return match.group(0).strip()

def extract_first_after_keyword(lines, keywords, default):
    for line in lines:
        line_lower = line.lower()
        for kw in keywords:
            if kw in line_lower:
                parts = line.split(":")
                if len(parts) > 1:
                    return parts[-1].strip()[:50]
    return default

def extract_first_non_empty_line(lines):
    for line in lines:
        line = line.strip()
        if line and len(line) > 2 and not any(c.isdigit() for c in line[:5]):
            return line[:50]
    return "N/A"

def count_keyword_occurrences(text, keywords):
    count = 0
    for kw in keywords:
        count += text.count(kw)
    return count

def count_unique_skills(text):
    skill_keywords = ["python", "java", "c++", "javascript", "react", "sql", "aws", "azure", "machine learning", "data science", "html", "css", "git", "docker", "kubernetes", "tensorflow", "pytorch"]
    found = [s for s in skill_keywords if s in text.lower()]
    return len(found)


@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    metadata = load_metadata()
    upload_docs = []

    for file in files:
        document_id = str(uuid.uuid4())
        extension = os.path.splitext(file.filename)[1]
        stored_filename = f"{document_id}{extension}"
        file_path = os.path.join(upload_dir, stored_filename)
        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        metadata[document_id] = {
            "document_id": document_id,
            "original_filename": file.filename,
            "stored_filename": stored_filename,
            "content_type": file.content_type,
            "path": file_path,
            "status": "uploaded"
        }

        upload_docs.append({
            "document_id": document_id,
            "filename": file.filename
        })
        
        save_metadata(metadata)

    chunk_metadata = load_chunks_metadata()
    current_size = 0
    
    if os.path.exists(faiss_file):
        index = faiss.read_index(faiss_file)
        current_size = index.ntotal
    else:
        index = None

    extracted_texts = {}
    extracted_info = {}

    for doc_info in upload_docs:
        doc_id = doc_info["document_id"]
        doc = load_document(doc_id)
        
        text = "\n".join(d.page_content for d in doc)
        extracted_texts[doc_id] = text
        
        doc_type = detect_document_type(text)
        extracted_info[doc_id] = extract_document_info(text, doc_type)
            
        chunks = chunk_document(text)
        embeddings = generate_embeddings(chunks)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        dimensions = embeddings.shape[1]

        if index is None:
            index = faiss.IndexFlatIP(dimensions)
        
        index.add(embeddings)

        for i, chunk in enumerate(chunks):
            chunk_metadata.append({
                "vector_id": current_size + i,
                "document_id": doc_id,
                "chunk_text": chunk
            })
        
        current_size += len(chunks)

        metadata = load_metadata()
        metadata[doc_id]["status"] = "indexed"
        metadata[doc_id]["doc_type"] = doc_type
        save_metadata(metadata)

    faiss.write_index(index, faiss_file)
    save_chunks_metadata(chunk_metadata)

    return {
        "status": "success",
        "document_ids": [doc["document_id"] for doc in upload_docs],
        "extracted_texts": extracted_texts,
        "extracted_info": extracted_info,
        "count": len(upload_docs)
    }


class GenerationRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/generate")
def generate_answer(request: GenerationRequest):
    query_embeddings = generate_embeddings([request.query])
    faiss.normalize_L2(query_embeddings)

    index = faiss.read_index(faiss_file)
    k_retrive = min(20, index.ntotal)
    scores, indices = index.search(query_embeddings, k_retrive)

    chunk_metadata = load_chunks_metadata()
    candidates = []

    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        
        item = chunk_metadata[int(idx)]
        candidates.append({
            "vector_id": int(idx),
            "chunk_text": item["chunk_text"],
            "document_id": item["document_id"],
            "retrieval_score": float(scores[0][i])
        })
        
    if not candidates:
        return {
            "query": request.query,
            "results": [],
            "message": "No chunks found"
        }
    
    candidate_text = [c["chunk_text"] for c in candidates]
    pairs = [[request.query, text] for text in candidate_text]
    rerank_scores = reranker.predict(pairs)
    
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(rerank_scores[i])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    final_results = candidates[:request.top_k]

    context = "\n\n".join(r["chunk_text"] for r in final_results)
    prompt = f"""Based on the following context, answer the question consicely.
    Context:
    {context}

    Question: {request.query}

    Answer:"""

    response = model_gen.generate_content(prompt)
    return {"answer": response.text}