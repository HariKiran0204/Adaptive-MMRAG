# 🧠 Adaptive-MMRAG: An Offline Multimodal Retrieval-Augmented Generation Framework for Privacy-Preserving Semantic Search

## A production-ready, research-grade multimodal RAG framework that ingests, indexes, and queries diverse data formats (PDF, DOCX, Images, Scanned Documents, Audio) with multi-agent adaptive decision-making — completely OFFLINE.

---

### 🎯 Research Positioning & Core Idea

Traditional Multimodal RAG systems rely on static, fixed retrieval policies (e.g. static Top-$K$, blind search across all modalities, fixed reranking). This leads to high latency on edge devices, irrelevant context retrieval, and model hallucinations when evidence is weak.

**Adaptive-MMRAG** introduces a specialized **5-Agent Adaptive Decision & Verification Layer** that dynamically analyzes query complexity, selects target modalities, allocates retrieval budgets, verifies evidence confidence, and triggers feedback-driven re-retrieval when initial evidence is insufficient — all while maintaining **100% privacy-preserving offline execution**.

---

### 🏛️ System Architecture

```
                                USER QUERY
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  Agent 1: Query Analysis     │
                     │  (Intent, Complexity, Modality)
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  Agent 2: Retrieval Planner  │
                     │  (Strategy, Top-K, Rerank)   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  Agent 3: Multimodal Search  │
                     │  (FAISS + BM25 + RRF + CLIP) │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ Agent 4: Evidence Verifier   │
                     │ (Confidence Score: 0.0 - 1.0)│
                     └──────────────┬───────────────┘
                                    │
                       ┌────────────┴────────────┐
                       │  Is Evidence Sufficient?│
                       └───┬─────────────────┬───┘
                      YES  │                 │ NO (Score < 0.50 & Iteration < 2)
                           │                 │
                           │                 ▼
                           │    ┌───────────────────────────┐
                           │    │ Adaptive Re-Retrieval     │
                           │    │ (Expand Top-K & Modality) │
                           │    └────────────┬──────────────┘
                           │                 │ (Re-search & Re-verify)
                           ▼                 ▼
                     ┌──────────────────────────────┐
                     │ Agent 5: Response Generator  │
                     │ (Qwen 2.5 3B / LLaVA Vision) │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                       GROUNDED STREAMING RESPONSE
                       + Expandable Adaptive Telemetry
```

---

### 🤖 The 5 Specialized Agents

1. **Query Analysis Agent (`QueryAnalysisAgent`)**:
   * Analyzes query intent (`factual`, `document_overview`, `comparison`, `visual_question`, `audio_question`, `cross_modal`, `reasoning`).
   * Evaluates computational complexity (`LOW`, `MEDIUM`, `HIGH`) in $<2\text{ms}$ using deterministic heuristics without LLM bottlenecks.
   * Selects required modality targets (`TEXT`, `IMAGE`, `AUDIO`, `TABLE`).

2. **Retrieval Planner Agent (`RetrievalPlannerAgent`)**:
   * Converts query analysis into a concrete, cost-aware `RetrievalPlan`.
   * Dynamically allocates Top-$K$ budget (e.g. $K=4$ for low-complexity lookups, $K=10$ for comparisons, $K=16$ for overviews).
   * Decides whether expensive Cross-Encoder reranking is needed.

3. **Multimodal Retrieval Agent (`MultimodalRetrievalAgent`)**:
   * Cleanly executes hybrid search using FAISS (dense 512-dim embeddings), BM25Okapi (lexical matching), Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.
   * Filters and partitions results according to target modalities.

4. **Evidence Verification Agent (`EvidenceVerificationAgent`)**:
   * Evaluates retrieved context and computes a normalized confidence score $S \in [0, 1]$ based on relevance scores, top-3 mean confidence, and context coverage.
   * Categorizes evidence sufficiency (`HIGH \ge 0.75`, `MEDIUM \ge 0.50`, `LOW < 0.50`).

5. **Response Generation Agent (`ResponseGenerationAgent`)**:
   * Formulates grounded, hallucination-aware prompts with strict verbatim constraints.
   * Streams tokens dynamically via local Ollama models (`qwen2.5:3b` / `llava:7b`).

---

### 🔄 Adaptive Re-Retrieval Feedback Loop

When initial evidence confidence is weak ($S < 0.50$):
1. The feedback loop automatically triggers an **Adaptive Re-Retrieval cycle** (up to `MAX_RETRIEVAL_ITERATIONS = 2`).
2. The planner expands the Top-$K$ budget by 50%, enables Cross-Encoder reranking, and broadens modality search scope.
3. The system re-verifies evidence and logs the score improvement in the telemetry trail.

---

### ⚙️ Baseline Mode vs. Adaptive Mode

The framework supports both execution modes for empirical benchmarking:
* **Adaptive Mode (Default)**: Full 5-Agent pipeline with dynamic Top-$K$, modality routing, verification, and re-retrieval.
* **Baseline Mode**: Traditional fixed-policy RAG (fixed $K=6$, static hybrid retrieval without verification loop).

Toggle easily via the web UI radio switch or request parameter: `{"query": "...", "mode": "baseline"}`.

---

### 🛡️ Privacy & Offline Guarantees

* **100% Local Inference**: Zero calls to external cloud APIs (no OpenAI, Gemini, or Claude).
* **Local Models**:
  * Text Model: `qwen2.5:3b` (via local Ollama)
  * Vision Model: `llava:7b` / `moondream` (via local Ollama)
  * OCR: Tesseract 5.0+ with local multi-language traineddata (`eng`, `tel`, etc.)
  * Speech-to-Text: `faster-whisper` base/int8 model running locally on CPU.
  * Embeddings: `sentence-transformers/clip-ViT-B-32` (512-dim shared text-image space).

---

### 🚀 Quickstart & Execution

#### 1. Start the System Offline
```powershell
# Run the automated offline startup script (sets offline flags and starts Ollama & Flask)
.\start_offline.ps1
```
The web application will be accessible at: `http://localhost:5000`

#### 2. Run the Verification Test Suite
```powershell
.\.venv\Scripts\python.exe test_adaptive_suite.py
```

#### 3. Generate Evaluation & Comparison Graphs
```powershell
.\.venv\Scripts\python.exe generate_evaluation_graphs.py
```
Generated graphs will be saved to `evaluation_graphs/`.

---

### 📊 Telemetry & Logging

Every query automatically logs comprehensive telemetry into `metrics/query_metrics.jsonl` and `metrics/query_metrics.csv`:
* `timestamp`, `session_id`, `mode`, `query`, `intent`, `complexity`
* `selected_modalities`, `retrieval_strategy`, `top_k`, `reranking`
* `model_name`, `model_tier`, `routing_reason`, `available_ram_gb`
* `initial_evidence_score`, `final_evidence_score`, `retrieval_iterations`, `re_retrieval`
* `claims_count`, `supported_claims_count`, `faithfulness_score`, `abstention_triggered`
* `retrieval_duration_seconds`, `generation_duration_seconds`, `total_query_duration_seconds`

---

### 🧪 Verification Guide: How to Verify Each New Component

#### 1. Verifying Post-Generation Evidence Verification & Abstention (Step 2)
* **How to trigger full abstention:**
  1. Upload a document (e.g. `recipe_book.pdf`).
  2. Ask a question completely outside the document corpus (e.g., *"What is the formula for calculating quantum entanglement in cryogenic semiconductors?"*).
  3. The system extracts claims, detects 0% claim support in evidence, and outputs the formal abstention notice:
     `"No supporting information was found in the available documents."`
  4. In `metrics/query_metrics.csv`, check that `abstention_triggered=True` and `faithfulness_score < 0.25`.

#### 2. Verifying Resource-Aware Multi-Model Routing (Step 3)
* **How to verify model tier selection:**
  1. Submit a simple lookup query (e.g., *"What is the applicant name?"*). Observe in the collapsible **Adaptive Details** card that `Tier 1 (qwen2.5:3b)` is routed.
  2. Submit a complex multi-document comparison query (e.g., *"Compare the architectural differences and revenue metrics across all uploaded files step by step."*).
  3. If available RAM >= 6.0 GB, observe that `Tier 2 (llama3.1:8b)` is dynamically routed for high-capacity reasoning.
  4. Check `metrics/query_metrics.csv` to see `model_tier`, `routing_reason`, and `available_ram_gb` logged for every query.

#### 3. Verifying Version-Aware Document Prioritization (Step 4)
* **How to verify version selection:**
  1. Upload two versions of a document (e.g., `policy_v1.pdf` and `policy_v2.pdf`).
  2. Query general policy: *"What is the current leave policy?"*. Observe that `policy_v2.pdf` chunks are prioritized.
  3. Query historical version: *"What was the leave policy in v1?"*. Observe that version-aware filtering respects the explicit version request.

#### 4. Running the Full Verification Test Suite
```powershell
python test_adaptive_suite.py
```
This executes all 10 automated unit and integration tests across query analysis, planning, verification, routing, post-generation claim checking, and version prioritization.

