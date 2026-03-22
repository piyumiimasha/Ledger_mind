
# Ledger_mind

Ledger_mind is a project focused on advanced financial document analysis, demonstrated on Uber Technologies' 2024 Annual Report. It features two main approaches: Retrieval-Augmented Generation (RAG) for question answering over documents, and parameter-efficient fine-tuning of large language models (LLMs) for domain adaptation.

---

## RAG (Retrieval-Augmented Generation)

**Purpose:**
RAG enables answering user questions by retrieving relevant information from a document collection and generating responses using an LLM. This approach combines the strengths of search and generative AI.

**Technology Stack:**
- **Vector Database:** Weaviate Embedded (runs locally, no server required)
- **Embeddings:** MiniLM-L6-v2 (384 dimensions)
- **Reranking:** Cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM:** Groq llama-3.1-8b-instant

**Why Weaviate Embedded?**
- Handles small to medium datasets (e.g., 464 chunks) efficiently
- No Docker or server setup needed—ideal for Windows and rapid prototyping
- Seamless migration to Weaviate Cloud (WCS) by changing only the client initialization

**Hybrid Search Pipeline:**
1. **Dense Retrieval:** Vector search using MiniLM-L6-v2 embeddings
2. **Sparse Retrieval:** BM25 keyword search (native in Weaviate)
3. **Fusion:** Combine dense and sparse results using Weaviate's hybrid search (alpha parameter controls weighting)
4. **Reranking:** Use a cross-encoder model to rerank the top results for better answer quality

**End-to-End Flow:**
1. User submits a question
2. The question is encoded into a vector (MiniLM-L6-v2)
3. Weaviate performs hybrid search (vector + BM25) to retrieve top 20 relevant chunks
4. Cross-encoder reranks these to select the top 3 most relevant chunks
5. The selected chunks are concatenated to form the context
6. The LLM (Groq llama-3.1-8b-instant) generates an answer, citing sources

---


## Fine-Tuning Large Language Models

**Purpose:**
Fine-tuning adapts a large pre-trained LLM (Mistral-7B) to answer questions specific to Uber's annual report, using a small set of Q&A pairs. This is done efficiently using quantization and LoRA adapters, so it can run on consumer GPUs.

**Step-by-Step Process:**
1. **Load Data:** 500 Q&A pairs about Uber's annual report
2. **Load Model:** Mistral-7B (7 billion parameters, via HuggingFace Transformers)
3. **Quantization:** Apply 4-bit quantization (bitsandbytes) to drastically reduce memory usage (from ~28GB to ~3.5GB)
4. **LoRA Adapters:** Inject LoRA adapters (via PEFT) to enable training only a small fraction (~1.1%) of the model parameters
5. **Fine-Tuning:** Train for 150 steps on the Q&A data
6. **Save Adapters:** Store the trained LoRA adapters (about 80MB)
7. **Inference:** Test the fine-tuned model on new questions

**Technology Stack:**
- **Model:** Mistral-7B (HuggingFace Transformers)
- **Quantization:** bitsandbytes (4-bit NF4 quantization)
- **Parameter-Efficient Fine-Tuning:** LoRA (PEFT library)
- **Training Framework:** HuggingFace Transformers + PEFT
- **Data Handling:** pandas, JSONL
- **Hardware:** Consumer GPU (8GB+ VRAM recommended)

**Key Libraries:**
- `transformers` — Model loading, training, and inference
- `bitsandbytes` — Efficient 4-bit quantization
- `peft` — Parameter-efficient fine-tuning (LoRA)
- `pandas` — Data manipulation
- `torch` — Core deep learning framework

**Why Quantization and LoRA?**
- Quantization allows running massive models on limited hardware by reducing memory requirements
- LoRA enables fast, low-cost fine-tuning by updating only a small subset of parameters

---

For more details, see the code and technical notes in this repository.

