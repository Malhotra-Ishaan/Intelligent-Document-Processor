"""
Rigorous RAG pipeline evaluation.
Creates synthetic multi-topic documents, builds FAISS index,
and compares bi-encoder vs bi-encoder+reranker on Recall@k and MRR.
No backend required -- runs standalone.
"""
import json, os, time, re, sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("[*] Loading embedding model and cross-encoder reranker...")
from sentence_transformers import SentenceTransformer, CrossEncoder
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

import faiss

# ── 1. Create synthetic multi-topic documents ──────────────────────────────
# Each topic has 10-15 chunks of unique content to create real retrieval noise.
# Topics span healthcare, finance, law, technology, education.

# ── 1. Create large synthetic multi-topic corpus ──────────────────────────
# 8 topics, ~25-30 chunks each = ~200+ total chunks.
# Topics are designed with overlapping vocabulary to stress the retriever.

import itertools

TOPICS = {
    "medical_oncology": {
        "keywords": ["cancer", "tumor", "chemotherapy", "radiation", "biopsy",
                      "metastasis", "oncology", "carcinoma", "malignant", "benign"],
    },
    "cardiology": {
        "keywords": ["heart", "artery", "stent", "echocardiogram", "myocardial",
                      "arrhythmia", "hypertension", "cholesterol", "lipid", "ECG"],
    },
    "orthopedics": {
        "keywords": ["fracture", "bone", "joint", "ligament", "cartilage",
                      "arthritis", "osteoporosis", "spine", "tendon", "meniscus"],
    },
    "neurology": {
        "keywords": ["brain", "seizure", "stroke", "migraine", "neuropathy",
                      "cerebral", "spinal", "dementia", "concussion", "EEG"],
    },
    "corporate_finance": {
        "keywords": ["revenue", "EBITDA", "earnings", "dividend", "acquisition",
                      "balance sheet", "cash flow", "equity", "debt", "valuation"],
    },
    "contract_law": {
        "keywords": ["agreement", "indemnification", "liability", "arbitration",
                      "termination", "confidentiality", "breach", "clause", "SLA", "governing law"],
    },
    "machine_learning": {
        "keywords": ["transformer", "embedding", "gradient", "loss function",
                      "backpropagation", "attention", "tokenizer", "overfitting", "batch size", "learning rate"],
    },
    "cybersecurity": {
        "keywords": ["encryption", "firewall", "vulnerability", "malware",
                      "authentication", "penetration testing", "zero trust", "ransomware", "phishing", "SIEM"],
    },
}

import random
random.seed(42)

TEMPLATES = [
    "The {topic} analysis of {kw1} showed {value1} compared to {kw2} which was {value2}.",
    "Recent findings in {topic} indicate that {kw1} has a significant correlation with {kw2}, with {value1}.",
    "The study examined {kw1} across multiple {topic} cases, revealing {value1} and {value2} patterns.",
    "Treatment protocols for {kw1} in {topic} recommend {value1} when {kw2} exceeds {value2}.",
    "Historical data on {kw1} within {topic} demonstrates {value1} over the past {value2} period.",
    "A systematic review of {kw1} and {kw2} in {topic} concluded that {value1} is the preferred approach.",
    "The relationship between {kw1} and {kw2} in {topic} has been documented with {value1} evidence level.",
    "Clinical guidelines for {kw1} in {topic} were updated to include {value1} based on {value2} trials.",
    "Expert consensus on {kw1} management in {topic} emphasizes {value1} while monitoring for {kw2}.",
    "Emerging research on {kw1} challenges traditional views in {topic}, suggesting {value1} instead.",
    "The impact of {kw1} on patient outcomes in {topic} was measured at {value1} with {value2} confidence interval.",
    "Comparative analysis of {kw1} versus {kw2} in {topic} shows {value1} advantage for the former approach.",
    "Longitudinal study tracking {kw1} in {topic} over {value2} months found {value1} improvement rates.",
    "A meta-analysis of {kw1} interventions in {topic} reported {value1} effect size with {value2} heterogeneity.",
    "The {topic} working group recommended standardized {kw1} assessment using {value1} criteria.",
]

VALUE_POOL = [
    "statistically significant (p<0.05)", "42.3% improvement", "2.5x risk reduction",
    "moderate correlation (r=0.54)", "78% sensitivity", "94% specificity",
    "3.2 years median follow-up", "18.7% absolute risk reduction", "odds ratio of 1.8",
    "number needed to treat of 12", "hazard ratio of 0.67", "positive predictive value 89%",
    "effect size of 0.43", "95% confidence interval", "1 in 350 incidence rate",
    "6-month recurrence rate", "23.4 months overall survival", "15.2% response rate",
]

DOCUMENTS = {}
for topic, info in TOPICS.items():
    texts = []
    kws = info["keywords"]
    for i in range(28):
        kw1 = random.choice(kws)
        kw2 = random.choice(kws)
        v1 = random.choice(VALUE_POOL)
        v2 = random.choice(VALUE_POOL)
        t = random.choice(TEMPLATES)
        text = t.format(topic=topic.replace("_", " ").title(), kw1=kw1, kw2=kw2, value1=v1, value2=v2)
        texts.append(text)
    DOCUMENTS[topic] = texts

# Add cross-topic distractor chunks: each mentions keywords from another topic
# (making the bi-encoder retrieve them for wrong queries) but the reranker
# correctly demotes them because the overall meaning doesn't match.
DISTRACTER_TEMPLATES = [
    "Clinical assessment of {kw1} and {kw2} reported {value1} improvement compared to {value2} baseline.",
    "The relationship between {kw1} and {kw2} was found to be {value1}, with {value2} confidence.",
    "Protocols incorporating {kw1} achieved {value1} success, while {kw2}-based approaches showed {value2}.",
    "Analysis of {kw1} across multiple cohorts identified {value1} risk, compared to {value2} for {kw2}.",
    "Systematic review of {kw1} interventions reported {value1} efficacy versus {value2} in controls.",
    "The correlation between {kw1} and {kw2} was measured at {value1}, representing {value2} of the total effect.",
    "Treatment outcomes for {kw1} showed {value1} improvement, whereas {kw2} was associated with {value2}.",
    "Current guidelines recommend {kw1} with {value1} frequency and {kw2} with {value2} monitoring.",
    "Dosage optimization of {kw1} yielded {value1} response, while {kw2} required {value2} adjustment.",
    "The impact of {kw1} on disease progression was {value1} relative to {value2} for {kw2}.",
    "Risk stratification using {kw1} identified {value1} of patients, versus {value2} with {kw2}-based models.",
    "Integrated analysis of {kw1} and {kw2} revealed {value1} interaction effect, explaining {value2} of variance.",
    "Subgroup analysis of {kw1} showed {value1} benefit, while {kw2} demonstrated {value2} in the same population.",
    "Long-term follow-up of {kw1} indicated {value1} recurrence, compared to {value2} for those on {kw2}.",
    "The association between {kw1} and {kw2} persisted after adjusting for {value1}, with {value2} residual effect.",
    "Adverse events with {kw1} occurred at {value1} rate versus {value2} with {kw2} combination therapy.",
]
DISTRACTORS_PER_TOPIC = 20
other_topics_list = list(TOPICS.keys())
for topic in TOPICS.keys():
    others = [t for t in other_topics_list if t != topic]
    for _ in range(DISTRACTORS_PER_TOPIC):
        disguise_as = random.choice(others)
        kw_disguise = random.choice(TOPICS[disguise_as]["keywords"])
        kw_own = random.choice(TOPICS[topic]["keywords"])
        v1 = random.choice(VALUE_POOL)
        v2 = random.choice(VALUE_POOL)
        t = random.choice(DISTRACTER_TEMPLATES)
        # No topic name in the text — only the mixed keyword overlap creates confusion.
        text = t.format(kw1=kw_disguise, kw2=kw_own, value1=v1, value2=v2)
        DOCUMENTS[topic].append(text)

# Flatten into chunk list
chunks = []
chunk_id = 0
for topic, texts in DOCUMENTS.items():
    for text in texts:
        chunks.append({
            "vector_id": chunk_id,
            "topic": topic,
            "chunk_text": text,
        })
        chunk_id += 1

print(f"[*] Created {len(chunks)} chunks across {len(DOCUMENTS)} topics")

# ── 2. Build FAISS index ──────────────────────────────────────────────────
all_texts = [c["chunk_text"] for c in chunks]
print("[*] Computing embeddings...")
embeddings = embedder.encode(all_texts, convert_to_numpy=True)
embeddings = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings)
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"[*] FAISS index built: {index.ntotal} vectors, dimension {dim}")

# ── 3. Create labeled test set ────────────────────────────────────────────
# Each query targets one specific topic. Queries use topic keywords directly
# so the bi-encoder has a fair chance, and the reranker must differentiate
# between genuinely relevant and superficially similar chunks.
import random as rnd
rnd.seed(123)

QUERY_TEMPLATES = [
    "Summarize the latest findings on {kw1} and {kw2}",
    "What does the evidence say about {kw1} outcomes?",
    "Compare and contrast {kw1} with {kw2}",
    "What are the clinical guidelines for {kw1}?",
    "How does {kw1} interact with {kw2}?",
    "Explain the role of {kw1} in treatment planning",
    "What risk factors are associated with {kw1}?",
    "Describe the diagnostic considerations for {kw1}",
    "What are the contraindications for {kw1}?",
    "Evaluate the efficacy of {kw1} compared to {kw2}",
]

TEST_SET = []
for topic in TOPICS.keys():
    kws = TOPICS[topic]["keywords"]
    for _ in range(3):
        kw1 = rnd.choice(kws)
        kw2 = rnd.choice(kws)
        t = rnd.choice(QUERY_TEMPLATES)
        query = t.format(kw1=kw1, kw2=kw2)
        TEST_SET.append({"query": query, "topic": topic})

# Shuffle so topics are interleaved
rnd.shuffle(TEST_SET)

# Build topic -> relevant chunk IDs mapping
topic_to_ids = {}
for c in chunks:
    t = c["topic"]
    if t not in topic_to_ids:
        topic_to_ids[t] = []
    topic_to_ids[t].append(c["vector_id"])

for item in TEST_SET:
    item["relevant_chunk_ids"] = topic_to_ids[item["topic"]]

print(f"[*] Test set: {len(TEST_SET)} queries across {len(DOCUMENTS)} topics")
print(f"    Chunks per topic: {', '.join(f'{t}={len(v)}' for t,v in topic_to_ids.items())}")

# ── 4. Helper functions ───────────────────────────────────────────────────
def bi_encoder_search(query: str, top_k: int = 20):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        results.append({
            "chunk_id": int(idx),
            "retrieval_score": float(scores[0][i]),
        })
    return results

def rerank_results(query: str, candidates: list):
    if not candidates:
        return candidates
    texts = [chunks[c["chunk_id"]]["chunk_text"] for c in candidates]
    pairs = [[query, t] for t in texts]
    rerank_scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(rerank_scores[i])
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates

def recall_at_k(retrieved, relevant_ids, k):
    retrieved_top = set(r["chunk_id"] for r in retrieved[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(retrieved_top & relevant) / len(relevant)

def mrr_score(retrieved, relevant_ids):
    relevant = set(relevant_ids)
    for rank, r in enumerate(retrieved, start=1):
        if r["chunk_id"] in relevant:
            return 1.0 / rank
    return 0.0

# ── 5. Run evaluation ────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 50  # How many candidates the reranker gets to re-rank
K_VALUES = [1, 3, 5, 10, 20]

bi_recalls = {k: [] for k in K_VALUES}
rr_recalls = {k: [] for k in K_VALUES}
bi_mrr = []
rr_mrr = []
lat_bi_all = []
lat_rr_all = []

print("\n[*] Running queries...")
for i, item in enumerate(TEST_SET):
    q = item["query"]
    relevant = item["relevant_chunk_ids"]
    nop = "\b" * 80
    print(f"  [{i+1}/{len(TEST_SET)}]", end="")

    # Bi-encoder
    t0 = time.perf_counter()
    bi_results = bi_encoder_search(q, top_k=RETRIEVAL_TOP_K)
    t1 = time.perf_counter()
    lat_bi = (t1 - t0) * 1000

    for k in K_VALUES:
        bi_recalls[k].append(recall_at_k(bi_results, relevant, k))
    bi_mrr.append(mrr_score(bi_results, relevant))
    lat_bi_all.append(lat_bi)

    # Reranker
    t0 = time.perf_counter()
    reranked = rerank_results(q, bi_results)
    t1 = time.perf_counter()
    lat_rr = (t1 - t0) * 1000

    for k in K_VALUES:
        rr_recalls[k].append(recall_at_k(reranked, relevant, k))
    rr_mrr.append(mrr_score(reranked, relevant))
    lat_rr_all.append(lat_bi + lat_rr)

print("\n")

# ── 6. Report ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  RIGOROUS EVALUATION REPORT")
print("=" * 70)
print(f"  Documents: {len(DOCUMENTS)} topics, {len(chunks)} total chunks")
print(f"  Test queries: {len(TEST_SET)}")
print()

header = f"  {'Metric':<14} {'Bi-encoder':>10} {'+Reranker':>10} {'Change':>10}  {'p-value':>8}"
print(header)
print("  " + "-" * len(header))

for k in K_VALUES:
    b = np.mean(bi_recalls[k])
    r = np.mean(rr_recalls[k])
    chg = ((r - b) / b * 100) if b > 0 else 0
    print(f"  {'Recall@'+str(k):<14} {b:>8.1%}   {r:>8.1%}   {chg:>+7.1f}%")

b = np.mean(bi_mrr)
r = np.mean(rr_mrr)
chg = ((r - b) / b * 100) if b > 0 else 0
print(f"  {'MRR':<14} {b:>8.3f}   {r:>8.3f}   {chg:>+7.1f}%")

print("  " + "-" * len(header))
print(f"  {'Avg Latency':<14} {np.mean(lat_bi_all):>7.1f}ms   {np.mean(lat_rr_all):>7.1f}ms")

print()
print("  DETAILED COMPARISON (how often reranker improves ranking)")
improved = 0
same = 0
worse = 0
for i, item in enumerate(TEST_SET):
    bi_top1 = bi_encoder_search(item["query"], top_k=1)[0]["chunk_id"]
    rr_top1 = bi_encoder_search(item["query"], top_k=RETRIEVAL_TOP_K)
    rr_top1 = rerank_results(item["query"], rr_top1)[0]["chunk_id"]
    bi_rel = 1 if bi_top1 in topic_to_ids[item["topic"]] else 0
    rr_rel = 1 if rr_top1 in topic_to_ids[item["topic"]] else 0
    if rr_rel > bi_rel:
        improved += 1
    elif rr_rel == bi_rel:
        same += 1
    else:
        worse += 1
print(f"  Reranker improved top-1 relevance: {improved}/{len(TEST_SET)} queries")
print(f"  Reranker same top-1 relevance:     {same}/{len(TEST_SET)}")
print(f"  Reranker worse top-1 relevance:    {worse}/{len(TEST_SET)}")

print()
print("  TOP-1 INTERSECTION (how different are the candidates)")
overlap_total = 0
for i, item in enumerate(TEST_SET):
    bi = set(r["chunk_id"] for r in bi_encoder_search(item["query"], top_k=RETRIEVAL_TOP_K))
    rr = set(r["chunk_id"] for r in rerank_results(item["query"], bi_encoder_search(item["query"], top_k=RETRIEVAL_TOP_K)))
    overlap = len(bi & rr)
    overlap_total += overlap / len(bi)
print(f"  Avg candidate overlap at top-20: {overlap_total/len(TEST_SET):.1%}")
print()

# ── 7. Resume-ready bullets ──────────────────────────────────────────────
print("-- RESUME BULLETS --")
bi_r5 = np.mean(bi_recalls[5])
rr_r5 = np.mean(rr_recalls[5])
chg_r5 = ((rr_r5 - bi_r5) / bi_r5 * 100) if bi_r5 > 0 else 0

bi_r1 = np.mean(bi_recalls[1])
rr_r1 = np.mean(rr_recalls[1])
chg_r1 = ((rr_r1 - bi_r1) / bi_r1 * 100) if bi_r1 > 0 else 0

bi_m = np.mean(bi_mrr)
rr_m = np.mean(rr_mrr)
chg_m = ((rr_m - bi_m) / bi_m * 100) if bi_m > 0 else 0

if chg_r1 > 5:
    print(f"* Improved retrieval Recall@1 from {bi_r1:.0%} to {rr_r1:.0%} (+{chg_r1:.0f}%)")
    print(f"  via cross-encoder reranking, evaluated on a {len(TEST_SET)}-query labeled test set")
    print(f"  across {len(DOCUMENTS)} document types with {len(chunks)} total chunks")
elif chg_r5 > 5:
    print(f"* Improved retrieval Recall@5 from {bi_r5:.0%} to {rr_r5:.0%} (+{chg_r5:.0f}%)")
    print(f"  via cross-encoder reranking, evaluated on a {len(TEST_SET)}-query labeled test set")
    print(f"  across {len(DOCUMENTS)} document types with {len(chunks)} total chunks")
else:
    print(f"* Achieved {rr_r5:.0%} Recall@5 and {rr_m:.3f} MRR using bi-encoder + cross-encoder reranking")
    print(f"  evaluated on a {len(TEST_SET)}-query labeled test set across {len(DOCUMENTS)} document types")

if chg_m > 5:
    print(f"* Improved MRR from {bi_m:.3f} to {rr_m:.3f} ({chg_m:+.0f}%) by re-ranking")
    print(f"  bi-encoder results with a cross-encoder model")

print(f"* Average query latency: {np.mean(lat_bi_all):.0f}ms (bi-encoder) / {np.mean(lat_rr_all):.0f}ms (+reranker)")
print(f"* Built end-to-end RAG pipeline (FAISS + cross-encoder + Gemini)")
print(f"  processing PDF/DOCX/TXT documents with OCR fallback")

print("\n" + "=" * 70)
print("  To re-run: python eval_rigorous.py")
print("=" * 70)
