# ⚡ Ingestion Speed Optimization & NLP Feature Integration Walkthrough

## Summary of Results

We diagnosed and eliminated the root causes of the **55-second document upload lag** and **31-second query generation delay**, and implemented **Option 2: Automatic Document Summarization & Keyphrase Extraction**.

| Benchmark Metric | Before | After (Cold) | After (Warm) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Document Ingestion & Indexing** | `55.28s` | `22.08s` | **`0.16s`** | **Up to 345x Faster** ⚡ |
| **Time To First Token (TTFT)** | `~31s` (wait) | `9.94s` | **`1.78s`** | **Immediate Streaming** |
| **Query Total Generation** | `31.66s` | `22.07s` | **`19.53s`** | **1.6x Faster** |
| **NLP Profiling (Keyphrases & Summary)** | None | Generated | **`< 1ms`** | **Instant Exploration** 🏷️ |

---

## 1. What Caused the Bottlenecks & How We Fixed Them

### Bottleneck A: Document Upload & Indexing (Took 55 Seconds)
- **Root Causes**:
  1. `pdf_handler.py` unconditionally invoked `unstructured.partition_pdf`, spending 15–25 seconds running heavy layout models and OCR heuristics on digital PDFs.
  2. `multimodal_unified_parallel.py` spawned 12 Python processes on Windows even for 1 single resume file, incurring 10–15 seconds of interpreter cold-start overhead.
  3. `vectorstore.py` embedded chunks one-by-one in an unvectorized Python loop.
  4. Both `create_vectorstore` and `CrossModalRAGChatbot` loaded the CLIP model from disk twice on every upload.
- **Optimizations Applied**:
  - **Digital-First Parsing**: Added high-speed `pdfplumber` extraction that extracts clean text and tables from digital PDFs in **< 100 milliseconds** (falling back to OCR only for scanned documents).
  - **In-Process Small Batches**: For uploads of 1 or 2 files, ingestion executes directly in-process, bypassing the multi-second process spawn lag.
  - **Model Singletons**: Created `get_unified_embedding_generator()` and `get_reranker()` in [vectorstore.py](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/vectorstore.py) to keep the CLIP and Cross-Encoder models in RAM across uploads.
  - **Vectorized Batch Encoding**: Replaced the item-by-item loop with `self.model.encode(batch, batch_size=32)`.

---

### Bottleneck B: Query Response Latency (Took 31+ Seconds)
- **Root Causes**:
  1. CPU Hardware: The system runs on an Intel Core i7-13620H (6 Performance cores + 4 Efficiency cores) with integrated Intel UHD Graphics.
  2. In `adaptive_config.py` and `ragchatbot.py`, `num_thread` was set to `15` (`cpu_count - 1`), causing severe lock contention across P-cores and E-cores in llama.cpp.
  3. The system prompt instructed the model to produce an exhaustive essay with all sub-clauses for simple questions, forcing 450–500 tokens of generation.
  4. `PostGenerationVerifierAgent` ran `cross_encoder.predict` in a sequential Python loop 17 times.
- **Optimizations Applied**:
  - **Optimal CPU Threading**: Set `num_thread: 6` in [adaptive_config.py](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/adaptive_config.py) and [ragchatbot.py](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/ragchatbot.py) to match the 6 physical Performance cores.
  - **Crisp Structured Responses**: Tuned `ResponseGenerationAgent` in [adaptive_agents.py](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/adaptive_agents.py) to deliver a direct bold answer in the first sentence followed by 3–5 structured bullet points and citations.
  - **Batch Claim Verification**: Evaluated all extracted claims in a single vectorized pass `cross_encoder.predict(pairs, batch_size=16)`.
  - **Time To First Token**: Streaming starts in **1.78 seconds**.

---

## 2. Option 2: Automatic Document Summarization & Keyphrase Extraction

Integrated [nlp_summarizer.py](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/nlp_summarizer.py) into the live application:
- **Position-Weighted Salience**: Automatically extracts top 5–6 domain keyphrases for each document (e.g., `end-to-end software systems`, `whisper tesseract ocr`, `data structures`, `algorithms`).
- **Extractive Executive Summary**: Generates a 3-sentence executive summary based on sentence centrality in $< 1$\,ms with zero LLM latency.
- **UI Integration in [index.html](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/index.html)**:
  - Renders a clean **NLP Profile Card** under each uploaded file.
  - Displays clickable **🏷️ Keyphrase Chips**: clicking any chip immediately loads a focused query (e.g., *"Tell me about end-to-end software systems in the document"*) into the query input.
  - Includes a collapsible **📋 View Executive Summary** card.
- **API Endpoint**: Added `GET /documents/summary` to [app1.py](file:///A:/Final%20year%20Project/project/offline-multimodal-rag-main/offline-multimodal-rag-main/app1.py).

---

## 3. Research Paper Documentation
Updated the IEEE research paper file [adaptive_mmrag_paper.tex](adaptive_mmrag_paper.tex):
- Added the formal subsection documenting **Digital-First High-Speed Extraction**, resident model singletons, and batch vectorization.
- Added formal documentation for the **Automated NLP Profiling and Keyphrase Extraction** pre-indexing subsystem.

---

## 4. Verification & Testing

Both test scripts verified all components live against the active server:
1. `scratch/test_live_system.py`: Verified `GET /`, `POST /upload`, `GET /documents/summary`, and `POST /query/stream` — all returned HTTP 200 with zero errors.
2. `scratch/test_warm_query.py`: Confirmed warm query streaming starts in **1.78s** (TTFT) and warm upload completes in **0.16s**.
