# Ledger_mind
Financial Analysis of Uber Technologies (2024 Annual Report) 


### RAG

##### Technology Stack
Vector Database: Weaviate Embedded (Python-only, no server needed)

Rationale: 464 chunks is small; embedded handles 100K+ objects easily
Benefit: No Docker/server management on Windows, faster for testing
Migration Path: Easy upgrade to Weaviate Cloud (WCS) by changing client init only


#### Hybrid Search Components:

Dense Retrieval: Vector search with MiniLM-L6-v2 embeddings (384 dims)
Sparse Retrieval: BM25 keyword search (Weaviate native, no external BM25 needed)
Fusion: Use Weaviate's built-in hybrid search with alpha parameter (dense_weight / total_weight)
Reranking: Cross-encoder model cross-encoder/ms-marco-MiniLM-L-6-v2

#### Pipeline Flow

User Question
  ↓
1. Encode Query (MiniLM-L6-v2)
  ↓
2. Weaviate Hybrid Search (Dense + BM25 with alpha=0.6)
   → Retrieve top 20 results
  ↓
3. Cross-Encoder Reranking (ms-marco-MiniLM-L6-v2)
   → Rerank to top 3 chunks
  ↓
4. Build Context (concatenate 3 chunks)
  ↓
5. LLM Generation (Groq llama-3.1-8b-instant)
   → Return answer + sources


