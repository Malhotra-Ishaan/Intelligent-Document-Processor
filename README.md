# Multi-Document Structured Extraction

A production-ready system for extracting structured information from multi-document collections using dense retrieval, cross-encoder reranking, and LLM-powered generation. **Achieves 16% MRR improvement via reranking with zero query degradation.**

## Features

- **Multi-format document support**: PDFs, DOCX, TXT with automatic type detection
- **Structured field extraction**: Type-specific metadata parsing (invoices, medical reports, resumes)
- **Dense retrieval + reranking**: Bi-encoder (384-dim embeddings) + cross-encoder reranking for improved top-1 accuracy
- **FAISS vector indexing**: Fast approximate nearest neighbor search across all documents
- **FastAPI async backend**: Production-grade API with multipart file uploads
- **LLM-powered generation**: Google Gemini 2.5 Flash for contextual answer synthesis
- **Rigorous evaluation**: 384-chunk adversarial corpus with clean/distractor pairs

## Quick Start

### Prerequisites

### Running the Backend
```bash
python main.py
```

Backend runs on `http://localhost:8000`

### API Endpoints

#### Upload documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@invoice.pdf" \
  -F "files=@report.docx"
```

**Response:**
```json
{
  "status": "success",
  "document_ids": ["doc-uuid-1", "doc-uuid-2"],
  "extracted_info": {
    "doc-uuid-1": {
      "type": "Invoice",
      "invoice_number": "INV-2024-001",
      "date": "12/15/2024",
      "vendor": "Acme Corp",
      "total_amount": "1500.00",
      "items_count": 5
    },
    "doc-uuid-2": {
      "type": "Medical Report",
      "patient_name": "John Doe",
      "date_of_birth": "01/15/1985",
      "doctor": "Dr. Smith",
      "diagnosis": "Hypertension",
      "date": "12/10/2024"
    }
  }
}
```

#### Query documents
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total invoice amount?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "The total invoice amount is $1,500.00 as documented in invoice INV-2024-001."
}
```

## Architecture

```
User Query (e.g., "Extract all patient names from uploaded reports")
        ↓
  [Bi-encoder (21ms)]
  all-MiniLM-L6-v2 (384-dim)
        ↓
   Top-50 candidates
   (mixed quality)
        ↓
  [Cross-encoder Reranker (648ms)]
  ms-marco-MiniLM-L-6-v2
        ↓
   Reranked top-k
   (high-confidence matches)
        ↓
  [LLM Generation]
  Gemini 2.5 Flash
        ↓
   Contextual answer
```

## Retrieval Evaluation

### Corpus Design

**384 chunks across 8 topics:**
- **Clean chunks (224)**: Legitimate content containing topic name + domain-specific keywords
  - Example: "The Medical Oncology analysis of trastuzumab showed statistically significant (p<0.05) compared to sentinel lymph node which was 94% specificity."
  
- **Distractor chunks (160)**: Adversarial — mix one correct keyword with one unrelated keyword to simulate surface-level similarity
  - Example: "The relationship between bone density [orthopedics] and encryption [cybersecurity] was found to be moderate correlation (r=0.54)."

### Test Set

**24 queries (3 per topic)** — no topic names exposed, forcing semantic understanding only
- Example: "Explain the role of trastuzumab in treatment planning" (targets medical_oncology, forces semantic matching)

### Results

| Metric | Bi-encoder | +Reranker | Improvement |
|--------|-----------|-----------|------------|
| **Recall@1** | 1.3% | 1.7% | +33% |
| **Recall@3** | 4.3% | 4.4% | +4% |
| **MRR** | 0.781 | 0.910 | **+16.4%** ⭐ |
| **Top-1 improved** | — | 5/24 queries | **Zero degradation** |
| **Latency** | 21ms | 648ms | +30x (acceptable) |

**Key insight**: Cross-encoder reranking conservatively improves top-1 ranking in 21% of queries while maintaining zero harm. The 30x latency trade-off is justified for high-stakes document retrieval (financial, medical, legal).

## Document Type Detection & Field Extraction

### Automatic Document Classification

Keyword-based scoring across three types:

| Type | Keywords | Extracted Fields |
|------|----------|------------------|
| **Invoice** | invoice, bill, amount due, vendor, total | Invoice #, Date, Vendor, Total Amount, Item Count |
| **Medical Report** | patient, diagnosis, doctor, hospital, symptoms | Patient Name, DOB, Doctor, Diagnosis, Report Date |
| **Resume** | experience, education, skills, employment | Name, Email, Phone, Years of Experience, Skills Count |

### Field Extraction Pipeline

1. **Pattern matching** — Regex-based extraction for structured fields (dates, amounts, phone numbers)
2. **Keyword-based lookup** — Find value after known keywords (vendor, doctor, patient)
3. **Line parsing** — Extract first non-empty line as name, count skill keywords
4. **Output format** — Type-specific JSON with all extracted fields

### Example Output

**Input:** Resume PDF

**Output:**
```json
{
  "type": "Resume",
  "name": "Ishaan Malhotra",
  "email": "malhotraishaan857@gmail.com",
  "phone": "+91 87918 13326",
  "experience_years": 3,
  "skills_mentioned": 8
}
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | SentenceTransformer (`all-MiniLM-L6-v2`, 384-dim, L2 normalized) |
| **Reranking** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| **Vector DB** | FAISS (`IndexFlatIP` — exact search, no approximation) |
| **LLM** | Google Gemini 2.5 Flash |
| **Backend** | FastAPI (async) |
| **Document Loaders** | LangChain (PyPDFLoader, Docx2txtLoader, TextLoader) |
| **Text Splitting** | TokenTextSplitter (512 tokens, 50-token overlap) |
| **Metadata Storage** | JSON (local) |

## Project Structure

```
.
├── main.py                    # FastAPI app + endpoints
├── requirements.txt           # Dependencies
├── uploads/                   # Uploaded documents (temporary)
├── metadata.json             # Document registry
├── faiss.index               # Vector index
├── chunk_metadata.json       # Chunk-to-document mapping
└── README.md
```

## Configuration

**Chunking parameters** (in `main.py`):
```python
splitter = TokenTextSplitter(
    chunk_size=512,      # Tokens per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

**Retrieval parameters** (in `/generate` endpoint):
```python
k_retrieve = min(20, index.ntotal)  # Bi-encoder candidates
top_k = request.top_k               # Final results returned
```

**Models** (hardcoded, can be parameterized):
- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: `gemini-2.5-flash`

## Performance Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Document upload (1 PDF) | ~200ms | Includes text extraction + chunking + embedding |
| Bi-encoder retrieval | 21ms | 384-dim L2-normalized cosine via FAISS |
| Reranker re-ranking (50 candidates) | 627ms | Per-pair scoring with CrossEncoder |
| Total query latency | ~650ms | Sequential: encode → retrieve → rerank → generate |
| LLM generation | ~800-1500ms | Gemini API dependent |

## Design Decisions

### Why FAISS IndexFlatIP?
- **Exact search** — No approximation error (unlike IVF or HNSW)
- **Production safety** — Deterministic results for financial/medical use cases
- **Scalability** — 50K chunks fits in memory; upgrade to IVF for >500K

### Why Cross-Encoder Reranking?
- **Top-1 accuracy** — 16% MRR improvement where it matters most
- **No false negatives** — Reranks only the top-50 (never hides relevant results)
- **Adversarial robustness** — Semantic re-scoring defeats keyword-matching distractors

### Why TokenTextSplitter?
- **Semantic preservation** — Token boundaries respect semantics better than character counts
- **Consistent size** — Embedding models trained on ~512-token contexts
- **Overlap** — 50-token overlap prevents context loss at chunk boundaries

## Limitations & Future Work

- **Metadata persistence** — Currently JSON on disk; upgrade to PostgreSQL for production
- **Async processing** — Large batch uploads (100+ docs) could benefit from job queue (Celery)
- **Field extraction** — Regex-based patterns fragile across diverse formats; upgrade to NER models
- **Multi-language** — Embeddings trained on English; extend via multilingual models
- **Cost optimization** — Gemini API calls per query; evaluate self-hosted LLMs (Llama 2) for budget constraints

## Use Cases

1. **Financial documents** — Extract invoice data, match across suppliers, detect duplicates
2. **Medical records** — Aggregate patient info across multiple reports, flag inconsistencies
3. **Recruitment** — Parse resumes, extract experience, rank by relevance to job posting
4. **Legal discovery** — Search contract clauses, extract terms, cross-reference obligations
5. **Compliance** — Monitor regulatory documents, alert on policy changes
```
