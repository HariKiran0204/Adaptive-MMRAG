# PDF Generator for Adaptive-MMRAG Research Journal Documentation
import os, subprocess

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Adaptive-MMRAG: Comprehensive Research Journal Specification</title>
<style>
    @page {
        size: A4;
        margin: 16mm 14mm 16mm 14mm;
        @bottom-right {
            content: 'Page ' counter(page);
            font-size: 8pt;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #64748b;
        }
        @bottom-left {
            content: 'Adaptive-MMRAG: Comprehensive Research Journal Specification';
            font-size: 8pt;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #64748b;
        }
    }
    body {
        font-family: 'Georgia', 'Cambria', serif;
        font-size: 9.6pt;
        line-height: 1.52;
        color: #1e293b;
        margin: 0;
        padding: 0;
        background-color: #ffffff;
    }
    .paper-header {
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 14px;
        border-bottom: 2px solid #0f172a;
    }
    h1.paper-title {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 18pt;
        font-weight: 700;
        line-height: 1.25;
        color: #0f172a;
        margin: 0 0 10px 0;
    }
    .paper-subtitle {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10pt;
        color: #2563eb;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .authors {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10pt;
        font-weight: 600;
        color: #334155;
        margin-bottom: 4px;
    }
    .affiliation {
        font-size: 8.8pt;
        color: #64748b;
        font-style: italic;
        margin-bottom: 10px;
    }
    .abstract-box {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        border-right: 1px solid #e2e8f0;
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 12px 16px;
        margin: 14px 0 18px 0;
        border-radius: 4px;
    }
    .abstract-title {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 700;
        font-size: 9.5pt;
        color: #0369a1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .abstract-text {
        font-size: 9pt;
        line-height: 1.48;
        text-align: justify;
        margin: 0;
    }
    .keywords {
        font-size: 8.6pt;
        margin-top: 8px;
        color: #334155;
    }
    .keywords strong {
        color: #0f172a;
    }
    h2 {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12pt;
        font-weight: 700;
        color: #0f172a;
        margin-top: 20px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        page-break-after: avoid;
    }
    h3 {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10.2pt;
        font-weight: 600;
        color: #1e3a8a;
        margin-top: 12px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }
    p {
        margin: 0 0 8px 0;
        text-align: justify;
    }
    ul, ol {
        margin: 0 0 10px 0;
        padding-left: 20px;
    }
    li {
        margin-bottom: 4px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 14px 0;
        font-size: 8.3pt;
        page-break-inside: avoid;
    }
    th, td {
        padding: 6px 8px;
        text-align: left;
        border: 1px solid #cbd5e1;
    }
    th {
        background-color: #0f172a;
        color: #ffffff;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 600;
        font-size: 8.3pt;
    }
    tr:nth-child(even) {
        background-color: #f8fafc;
    }
    .table-caption {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 8.3pt;
        font-weight: 600;
        color: #475569;
        margin-bottom: 4px;
        text-align: left;
    }
    .math-block {
        background-color: #f1f5f9;
        border-left: 3px solid #64748b;
        padding: 6px 12px;
        margin: 8px 0;
        font-family: 'Consolas', monospace;
        font-size: 8.8pt;
        color: #0f172a;
        border-radius: 2px;
        page-break-inside: avoid;
    }
    .math-formula {
        font-family: 'Cambria Math', 'Georgia', serif;
        font-style: italic;
        font-weight: 600;
        color: #0f172a;
    }
    .arch-diagram {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 10px;
        border-radius: 4px;
        font-family: 'Consolas', monospace;
        font-size: 7.6pt;
        line-height: 1.3;
        white-space: pre;
        margin: 10px 0;
        color: #0f172a;
        page-break-inside: avoid;
    }
    .highlight-box {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 8px 12px;
        margin: 10px 0;
        border-radius: 3px;
        font-size: 8.8pt;
        page-break-inside: avoid;
    }
    .highlight-title {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #047857;
        margin-bottom: 4px;
    }
    .ref-list {
        font-size: 8pt;
        line-height: 1.35;
        padding-left: 18px;
    }
    .ref-list li {
        margin-bottom: 4px;
    }
</style>
</head>
<body>

<div class="paper-header">
    <div class="paper-subtitle">Comprehensive Research Journal Specification & Architecture Document</div>
    <h1 class="paper-title">Adaptive-MMRAG: A Multi-Agent, Resource-Aware, Offline Multimodal Retrieval-Augmented Generation Framework with Dual-Stage Evidence Verification</h1>
    <div class="authors">Abhinay Sivva, Vankadaru Hari Kiran, Adithya Gaddam, Yashwanth</div>
    <div class="affiliation">Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Amaravati Campus, Andhra Pradesh, India</div>
</div>

<div class="abstract-box">
    <div class="abstract-title">Abstract</div>
    <p class="abstract-text">
        Standard offline Retrieval-Augmented Generation (RAG) systems suffer from rigid, one-size-fits-all retrieval policies that map every user query through static top-K thresholds, hardcode a single local language model, and lack automated factual grounding verification. When deployed on resource-constrained consumer hardware, this design induces severe factual hallucination rates (up to 28.0%), memory pressure causing out-of-memory (OOM) failures, and unresponsiveness. This document presents the complete architectural specification and empirical findings of <strong>Adaptive-MMRAG</strong>, a fully offline, zero-cloud multimodal framework designed for commodity edge machines. Adaptive-MMRAG introduces: (1) a deterministic six-agent decision pipeline that performs sub-2ms query intent and complexity classification without LLM overhead; (2) a real-time, resource-aware multi-model router tracking physical host RAM via <code>psutil</code> that switches between lightweight (3B) and deep reasoning (8B) models to guarantee OOM-free execution; (3) a dual-stage evidence verification mechanism coupling pre-generation confidence scoring (S in [0, 1]) with post-generation factual claim auditing (F in [0, 1]) and a calibrated three-tier abstention gate; (4) a high-speed digital-first ingestion engine that achieves up to 345x faster document indexing (0.16s warm); and (5) an offline NLP profiling engine delivering instant position-weighted TF-IDF keyphrase extraction and sentence-centrality summarization. Across a 100-query benchmark, Adaptive-MMRAG reduces factual hallucination from 28.0% to 2.5%, elevates faithfulness from 0.68 to 0.96, cuts peak RAM by 45.9% (8.4 GB to 4.7 GB), and achieves 94.2% abstention accuracy on unanswerable queries—operating with zero cloud API dependencies.
    </p>
    <div class="keywords">
        <strong>Keywords:</strong> Retrieval-Augmented Generation, Offline Inference, Multi-Agent Systems, Multimodal Information Retrieval, Edge AI, Hallucination Mitigation, Resource-Aware Routing, Dual-Stage Verification.
    </div>
</div>

<h2>1. Introduction & Research Problem Formulation</h2>
<p>
    Retrieval-Augmented Generation (RAG) has emerged as the premier methodology for mitigating the inherent hallucination tendencies of Large Language Models (LLMs) by grounding generation in domain-specific corpora. However, enterprise, healthcare, and defence applications frequently mandate <strong>strict data sovereignty and air-gapped offline operation</strong>, prohibiting reliance on cloud-hosted proprietary models (e.g., GPT-4V, Claude, Gemini). Deploying RAG systems purely on edge consumer hardware exposes three critical structural bottlenecks:
</p>
<ul>
    <li><strong>Static Retrieval Policy:</strong> Conventional pipelines retrieve an invariant number of top-K passages (typically K=4 to K=6) regardless of whether the query is a simple factual lookup or an exhaustive multi-document synthesis. This induces severe context dilution on simple queries and information starvation on complex ones.</li>
    <li><strong>Static Single-Model Failure Modes:</strong> Hardcoding a single local LLM creates a fatal tradeoff: lightweight models (&le; 3B parameters) exhibit reasoning degradation on complex queries, while larger models (7B&ndash;8B parameters) exhaust host RAM (triggering Windows swap thrashing or OOM crashes) when background workloads fluctuate.</li>
    <li><strong>The Verification Void:</strong> Traditional RAG scores passage relevance <em>prior</em> to generation, but performs no audit of the tokens actually emitted by the LLM. Consequently, models frequently fabricate unsupported assertions despite possessing high-quality retrieval context.</li>
</ul>
<p>
    <strong>Adaptive-MMRAG</strong> resolves these challenges through an end-to-end multi-agent orchestration architecture that dynamically modulates retrieval depth, inspects host hardware state before model dispatch, and verifies every generated proposition against source evidence before presenting answers to the user.
</p>

<h2>2. End-to-End System Architecture</h2>
<p>
    Adaptive-MMRAG is structured into three integrated subsystems: (i) the Multimodal Ingestion Engine, (ii) the Six-Agent Orchestration Core, and (iii) the Dual-Stage Verification & Gatekeeper Layer.
</p>

<div class="arch-diagram">
+----------------------------------------------------------------------------------------------------+
|                                    USER INTERFACE / VOICE INPUT                                    |
+----------------------------------------------------------------------------------------------------+
       |                                                                               ^
       v                                                                               |
+------------------------------+     +-------------------------------+     +-------------------------+
| Agent 1: Query Analysis      | --> | Agent 2: Retrieval Planner    | --> | Agent 3: Hybrid Search  |
| - Intent (9 classes)         |     | - Dynamic Top-K (4 to 12)     |     | - Dense: FAISS (CLIP)   |
| - Complexity (LOW/MED/HIGH)  |     | - Strategy: Hybrid / Semantic |     | - Sparse: BM25Okapi     |
| - Target Modalities          |     | - Cross-Encoder Flag          |     | - Reciprocal Rank Fusion|
+------------------------------+     +-------------------------------+     +-------------------------+
                                                                                       |
                                                                                       v
+----------------------------------------------------------------------------------------------------+
| Agent 4: Pre-Generation Evidence Verifier (Computes Confidence Score S in [0, 1])                  |
| - If S &lt; 0.50 and iter &lt; 2: Trigger Adaptive Re-Retrieval (Expand K by 1.4x, Broaden Modality) |
| - If S &gt;= 0.50: Proceed to Deterministic Resource-Aware Model Router                               |
+----------------------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------------------+
| Resource-Aware Model Router (Real-time psutil RAM Telemetry Check: Threshold theta = 6.0 GB)       |
| - RAM &lt; 6.0 GB OR Low Complexity  ==&gt; Tier 1 Model (Qwen2.5:3b / Granite-3b) [OOM Prevention]     |
| - RAM &gt;= 6.0 GB AND High Complexity ==&gt; Tier 2 Model (Llama3.1:8b / DeepSeek-R1) [Deep Reasoning]  |
+----------------------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------------------+
| Agent 5: Grounded Response Generation (Strict Prompt Constraints, CPU Pinning = 6 P-Cores)         |
+----------------------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------------------+
| Agent 6: Post-Generation Claim Verifier (Calculates Faithfulness Score F in [0, 1])                |
| - Sentence claim extraction + Non-stopword lexical overlap (rho &gt;= 0.35)                            |
| - Numeric / temporal invariance check (100% strict verification)                                   |
| - Cross-Encoder NLI entailment check (logit &gt;= -2.0)                                               |
+----------------------------------------------------------------------------------------------------+
                                           |
       +-----------------------------------+-----------------------------------+
       | (F &gt;= 0.60)                       | (0.25 &lt;= F &lt; 0.60)                | (F &lt; 0.25)
       v                                   v                                   v
+-----------------------------+     +-----------------------------+     +-----------------------------+
| Fully Grounded Response     |     | Response + Grounding Warning|     | Formal Abstention Message   |
| (Pass verbatim + Citations) |     | (Flags unverified claims)   |     | ('No supporting info found')|
+-----------------------------+     +-----------------------------+     +-----------------------------+
</div>

<h2>3. Mathematical Formulations & Algorithmic Principles</h2>

<h3>3.1. Multimodal Hybrid Retrieval and Rank Fusion</h3>
<p>
    Given a user query q, dense semantic search projects q into the 512-dimensional CLIP vector space:
</p>
<div class="math-block">
    <span class="math-formula">e_q = CLIP(q) / ||CLIP(q)||_2, &nbsp;&nbsp;&nbsp; Sim_dense(q, d) = e_q &bull; e_d</span>
</div>
<p>
    Concurrently, BM25Okapi sparse lexical scoring calculates exact keyword matching over tokenized text chunks:
</p>
<div class="math-block">
    <span class="math-formula">Score_BM25(q, d) = &sum;_{t &isin; q} IDF(t) &bull; [ f(t, d) &bull; (k_1 + 1) ] / [ f(t, d) + k_1 &bull; (1 - b + b &bull; (|d| / avgdl)) ]</span>
</div>
<p>
    where k_1 = 1.5 and b = 0.75. To merge disjoint score distributions without scale calibration bias, candidates are merged via <strong>Reciprocal Rank Fusion (RRF)</strong> with smoothing constant k_rrf = 60:
</p>
<div class="math-block">
    <span class="math-formula">RRF(d) = &sum;_{L &isin; {L_dense, L_bm25}} [ 1 / (60 + r_L(d)) ]</span>
</div>
<p>
    The top-N candidates (N=10) are reranked via a cross-attention transformer (<code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>) evaluated across batches (B=16):
</p>
<div class="math-block">
    <span class="math-formula">s_ce(d) = Softmax( W &bull; Transformer([q || SEP || d]) )</span>
</div>

<h3>3.2. Pre-Generation Retrieval Confidence Scoring (S)</h3>
<p>
    Agent 4 computes an objective confidence metric S in [0, 1] before passing context to the LLM:
</p>
<div class="math-block">
    <span class="math-formula">S = &alpha; &bull; r_3_bar + &beta; &bull; c_cover + &gamma; &bull; c_kw</span>
</div>
<p>
    where &alpha; = 0.5, &beta; = 0.3, &gamma; = 0.2. r_3_bar represents the sigmoid-normalized mean logit of the top-3 reranked chunks, c_cover measures target modality representation, and c_kw calculates the lexical recall of query content keywords within the retrieved passages.
</p>

<h3>3.3. Resource-Aware Dynamic Model Routing Policy</h3>
<p>
    The router queries physical system memory telemetry RAM_avail = psutil.virtual_memory().available / 2^30 and dispatches models via the deterministic decision rule:
</p>
<div class="math-block">
    <span class="math-formula">
    M(RAM_avail, C_q, E_S) = <br/>
    &nbsp;&nbsp;&bull; Tier 1 (Qwen2.5:3b), &nbsp; if RAM_avail &lt; 6.0 GB &nbsp; [OOM Prevention]<br/>
    &nbsp;&nbsp;&bull; Tier 1 (Qwen2.5:3b), &nbsp; if C_q = LOW and E_S &isin; {HIGH, MED} &nbsp; [Fast Path]<br/>
    &nbsp;&nbsp;&bull; Tier 2 (Llama3.1:8b), &nbsp; if C_q &isin; {HIGH, OVERVIEW} and RAM_avail &ge; 6.0 GB &nbsp; [Deep Reasoning]<br/>
    &nbsp;&nbsp;&bull; Tier 1 (Qwen2.5:3b), &nbsp; otherwise &nbsp; [Conservative Path]
    </span>
</div>

<h3>3.4. Post-Generation Factual Claim Verification & Abstention</h3>
<p>
    Agent 6 parses the generated response into discrete propositional assertions C = {c_1, ..., c_m}. Each proposition is validated across three orthogonal criteria:
</p>
<ol>
    <li><strong>Lexical Support:</strong> Overlap ratio &rho;_i = |W(c_i) &cap; W(E)| / |W(c_i)| &ge; 0.35, where E is the retrieved context.</li>
    <li><strong>Numeric & Temporal Invariance:</strong> Strict condition that all numbers in c_i must appear in E.</li>
    <li><strong>Cross-Encoder Entailment:</strong> Reranker NLI logit CE_logit(c_i, E) &ge; -2.0.</li>
</ol>
<p>
    The composite Faithfulness Score F is defined as the fraction of supported claims:
</p>
<div class="math-block">
    <span class="math-formula">F = [ &sum;_{i=1}^{m} I(c_i is supported) ] / m</span>
</div>
<p>
    The output is governed by a calibrated three-tier policy:
</p>
<ul>
    <li>F &ge; 0.60: Answer passed verbatim with verified chunk citations.</li>
    <li>0.25 &le; F &lt; 0.60: Grounding warning prepended: <em>"[Warning: Some assertions lack direct evidence in the source documents]"</em>.</li>
    <li>F &lt; 0.25: Output suppressed and formal abstention emitted: <em>"No supporting information was found in the available documents."</em></li>
</ul>

<h2>4. Multimodal Ingestion Pipeline & High-Speed Optimizations</h2>
<p>
    The ingestion engine supports five document formats entirely on-device without internet access:
</p>
<table>
    <thead>
        <tr>
            <th>Document Format</th>
            <th>Primary Extraction Engine</th>
            <th>Fallback / Processing Method</th>
            <th>Latent Representation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Text PDF</strong></td>
            <td><code>pdfplumber</code> native extraction (&lt;100ms)</td>
            <td>Structured table grid reconstruction</td>
            <td>512-d CLIP Vector + BM25 Lexical</td>
        </tr>
        <tr>
            <td><strong>Scanned PDF</strong></td>
            <td>Character count check (&lt;50 chars)</td>
            <td><code>pypdfium2</code> render + Tesseract 5.x OCR</td>
            <td>512-d CLIP Vector + BM25 Lexical</td>
        </tr>
        <tr>
            <td><strong>DOCX Document</strong></td>
            <td><code>python-docx</code> paragraph parser</td>
            <td>Table cell serialization + image dump</td>
            <td>512-d CLIP Vector + BM25 Lexical</td>
        </tr>
        <tr>
            <td><strong>Standalone Images</strong></td>
            <td>CLIP Vision ViT-B-32 Encoder</td>
            <td>Tesseract OCR text extraction</td>
            <td>512-d CLIP Joint Visual-Text Space</td>
        </tr>
        <tr>
            <td><strong>Audio Transcripts</strong></td>
            <td>OpenAI Whisper Base (Int8 CPU)</td>
            <td>Timestamped chunking with overlap</td>
            <td>512-d CLIP Vector + BM25 Lexical</td>
        </tr>
    </tbody>
</table>

<div class="highlight-box">
    <div class="highlight-title">Engineering Solutions for Ingestion & Latency Bottlenecks</div>
    <ul>
        <li><strong>Digital-First Bypass:</strong> Replaced the heavy, multi-model <code>unstructured</code> pipeline with native <code>pdfplumber</code> extraction for text PDFs, dropping parse times from 25 seconds to under 100 milliseconds.</li>
        <li><strong>Windows ProcessPool Elimination:</strong> Ingestion of &le; 2 files is handled strictly in-process, eliminating the 10&ndash;15 second interpreter spawn overhead inherent to Windows multiprocessing.</li>
        <li><strong>Resident RAM Singletons:</strong> Embedder and reranker model instances are retained persistently in RAM across requests, eliminating repeated disk I/O.</li>
        <li><strong>Vectorized Batched Inference:</strong> Chunk vectorization is batched (B=32), achieving an 80% reduction in embedding generation latency.</li>
        <li><strong>Performance Core Pinning:</strong> Ollama threads are pinned to 6 physical Performance cores (P-cores) on the Intel Core i7-13620H, resolving cross-core locking contention between P-cores and E-cores and dropping TTFT to 1.78s.</li>
    </ul>
</div>

<h2>5. Automated Document-Level NLP Profiling</h2>
<p>
    Implemented in <code>nlp_summarizer.py</code>, the NLP engine runs synchronously alongside indexing in under 15 milliseconds without LLM calls:
</p>
<ul>
    <li><strong>Position-Weighted TF-IDF Keyphrase Extraction:</strong> Ranks domain keyphrases by combining term frequency, inverted document frequency, positional salience (rewarding terms in headings and opening paragraphs), and multi-word bigram/trigram collocation scoring. Keyphrases are rendered in the web UI as clickable chips to enable instant query generation.</li>
    <li><strong>Extractive Centrality Summarization:</strong> Decomposes documents into sentence graphs, evaluates salience density, and compiles a concise 3-sentence executive summary with zero LLM compute overhead.</li>
</ul>

<h2>6. Empirical Evaluation & Benchmark Results</h2>

<div class="table-caption">Table 1: Cumulative 5-Arm Ablation Study Benchmark across 100 Evaluation Queries.</div>
<table>
    <thead>
        <tr>
            <th>Experimental Configuration</th>
            <th>Recall@5</th>
            <th>Faithfulness (F)</th>
            <th>Hallucination Rate</th>
            <th>Mean Latency</th>
            <th>Peak RAM</th>
            <th>Abstention Acc.</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Arm 1: Baseline (Static RAG, K=6)</strong></td>
            <td>0.72</td>
            <td>0.68</td>
            <td>28.0%</td>
            <td>4.85 s</td>
            <td>8.4 GB</td>
            <td>0.0%</td>
        </tr>
        <tr>
            <td><strong>Arm 2: + Hybrid Search & Reranking</strong></td>
            <td>0.88</td>
            <td>0.74</td>
            <td>21.0%</td>
            <td>3.42 s</td>
            <td>8.5 GB</td>
            <td>0.0%</td>
        </tr>
        <tr>
            <td><strong>Arm 3: + Adaptive Retrieval Loop (Agent 4)</strong></td>
            <td>0.94</td>
            <td>0.81</td>
            <td>14.0%</td>
            <td>2.85 s</td>
            <td>8.5 GB</td>
            <td>0.0%</td>
        </tr>
        <tr>
            <td><strong>Arm 4: + Resource-Aware Model Router</strong></td>
            <td>0.94</td>
            <td>0.85</td>
            <td>11.0%</td>
            <td><strong>1.95 s</strong></td>
            <td><strong>4.6 GB</strong></td>
            <td>0.0%</td>
        </tr>
        <tr>
            <td><strong>Arm 5: Full Adaptive-MMRAG (+ Agent 6 Verifier)</strong></td>
            <td><strong>0.94</strong></td>
            <td><strong>0.96</strong></td>
            <td><strong>2.5%</strong></td>
            <td>2.10 s</td>
            <td>4.7 GB</td>
            <td><strong>94.2%</strong></td>
        </tr>
    </tbody>
</table>

<div class="table-caption">Table 2: Live Ingestion & Inference Latency Optimization Benchmarks (Evaluated on Technical Document).</div>
<table>
    <thead>
        <tr>
            <th>Evaluation Metric</th>
            <th>Prior to Optimization</th>
            <th>Optimized (Cold Start)</th>
            <th>Optimized (Warm Cache)</th>
            <th>Overall Speedup</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Document Ingestion & Indexing</strong></td>
            <td>55.28 s</td>
            <td>22.08 s</td>
            <td><strong>0.16 s</strong></td>
            <td><strong>Up to 345x Faster &#9889;</strong></td>
        </tr>
        <tr>
            <td><strong>Time To First Token (TTFT)</strong></td>
            <td>~31.0 s (Blocking)</td>
            <td>9.94 s</td>
            <td><strong>1.78 s</strong></td>
            <td><strong>Immediate Token Stream</strong></td>
        </tr>
        <tr>
            <td><strong>Total Query Generation Time</strong></td>
            <td>31.66 s</td>
            <td>22.07 s</td>
            <td><strong>19.53 s</strong></td>
            <td><strong>1.62x Faster</strong></td>
        </tr>
        <tr>
            <td><strong>NLP Document Profiling & Keyphrases</strong></td>
            <td>Not available</td>
            <td>&lt; 15 ms</td>
            <td><strong>&lt; 1 ms</strong></td>
            <td><strong>Instantaneous</strong></td>
        </tr>
    </tbody>
</table>

<div class="table-caption">Table 3: Modality-Level Semantic Similarity in Joint CLIP 512-d Embedding Space.</div>
<table>
    <thead>
        <tr>
            <th>Target Document Modality</th>
            <th>Mean Semantic Similarity (cos &theta;)</th>
            <th>Active Processing Subsystem</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Digital Text / PDF</strong></td>
            <td><strong>0.91</strong></td>
            <td><code>pdfplumber</code> high-speed text extraction</td>
        </tr>
        <tr>
            <td><strong>DOCX Structured Tables</strong></td>
            <td><strong>0.89</strong></td>
            <td><code>python-docx</code> structured XML parsing</td>
        </tr>
        <tr>
            <td><strong>Cross-Modal Hybrid Fusion</strong></td>
            <td><strong>0.86</strong></td>
            <td>Joint CLIP Visual-Text Vector Alignment</td>
        </tr>
        <tr>
            <td><strong>Scanned Document OCR</strong></td>
            <td><strong>0.82</strong></td>
            <td><code>pypdfium2</code> + Tesseract 5.x LSTM OCR</td>
        </tr>
        <tr>
            <td><strong>Spoken Voice Transcripts</strong></td>
            <td><strong>0.78</strong></td>
            <td>OpenAI Whisper Base (Local CPU Execution)</td>
        </tr>
    </tbody>
</table>

<h2>7. Codebase Directory Mapping & Module Responsibilities</h2>
<p>
    The complete system resides on the <code>A:</code> drive in <code>A:\\Final year Project\\project\\offline-multimodal-rag-main\\offline-multimodal-rag-main</code>:
</p>
<ul>
    <li><strong><code>app1.py</code>:</strong> Core Flask web application, REST API endpoints, Server-Sent Events (SSE) token streaming, and document upload orchestrator.</li>
    <li><strong><code>index.html</code>:</strong> Responsive web client featuring real-time token rendering, clickable &#127991;&#65039; Keyphrase Chips, collapsible executive summary cards, and audio controls.</li>
    <li><strong><code>adaptive_agents.py</code>:</strong> Complete implementation of all 6 agents (Query Analysis, Retrieval Planning, Multimodal Search, Evidence Verifier, Response Generator, and Post-Gen Claim Verifier).</li>
    <li><strong><code>adaptive_controller.py</code>:</strong> Central orchestration controller managing multi-agent pipeline state, telemetry logging, and re-retrieval cycles.</li>
    <li><strong><code>adaptive_config.py</code>:</strong> Configuration hyperparameters, resource thresholds (&theta;_ram = 6.0 GB), and CPU thread tuning (pinned to 6 P-cores).</li>
    <li><strong><code>model_router.py</code>:</strong> <code>ResourceAwareModelRouter</code> tracking physical host memory and CPU utilization via <code>psutil</code>.</li>
    <li><strong><code>vectorstore.py</code>:</strong> Multimodal FAISS vector database (<code>IndexFlatIP</code>), BM25 lexical index, resident CLIP singleton, and Cross-Encoder reranker.</li>
    <li><strong><code>nlp_summarizer.py</code>:</strong> Pure Python offline NLP profiling engine for TF-IDF keyphrase extraction and extractive graph summarization.</li>
    <li><strong><code>pdf_handler.py</code>:</strong> Digital-first PDF parser leveraging <code>pdfplumber</code> with automatic fallback to <code>pypdfium2</code> and Tesseract OCR.</li>
    <li><strong><code>doc_handler.py</code>:</strong> Structured DOCX parser extracting tables, text blocks, and embedded figures.</li>
    <li><strong><code>image_handler.py</code>:</strong> Image ingestion pipeline executing Tesseract OCR and CLIP vision vector encoding.</li>
    <li><strong><code>audio_handler.py</code>:</strong> Local speech-to-text pipeline powered by quantized OpenAI Whisper.</li>
    <li><strong><code>tts_handler.py</code>:</strong> Offline text-to-speech synthesis using <code>pyttsx3</code> with regex-based markdown/citation sanitization.</li>
    <li><strong><code>start_offline.ps1</code>:</strong> Production PowerShell startup script enforcing air-gapped offline environment variables and background Ollama service launch.</li>
    <li><strong><code>adaptive_mmrag_paper.tex</code>:</strong> Complete IEEE conference LaTeX manuscript containing all formal mathematics and benchmark figures.</li>
</ul>

<h2>8. Limitations & Future Research Directions</h2>
<ul>
    <li><strong>Historical and Degraded Scans:</strong> Tesseract 5.x accuracy degrades on non-standard typography or damaged historical scans. Future work will investigate embedding an offline lightweight vision-language OCR model (e.g., PaddleOCR).</li>
    <li><strong>Audio Acoustic Noise:</strong> Whisper base transcription noise can propagate into vector embeddings; integrating speaker diarization (e.g., PyAnnote) and acoustic denoising will improve audio retrieval fidelity.</li>
    <li><strong>Semantic Claim Decomposition:</strong> Agent 6 currently employs sentence-boundary regex rules; replacing this with an offline transformer-based semantic proposition segmenter will further improve claim audit precision.</li>
</ul>

<h2>9. References</h2>
<ol class="ref-list">
    <li>P. Lewis et al., &ldquo;Retrieval-augmented generation for knowledge-intensive NLP tasks,&rdquo; in <em>Proc. NeurIPS</em>, vol. 33, pp. 9459&ndash;9474, 2020.</li>
    <li>A. Radford et al., &ldquo;Learning transferable visual models from natural language supervision,&rdquo; in <em>Proc. ICML</em>, pp. 8748&ndash;8763, 2021.</li>
    <li>A. Radford et al., &ldquo;Robust speech recognition via large-scale weak supervision,&rdquo; in <em>Proc. ICML</em>, pp. 28492&ndash;28518, 2023.</li>
    <li>J. Johnson, M. Douze, and H. J&eacute;gou, &ldquo;Billion-scale similarity search with GPUs,&rdquo; <em>IEEE Trans. Big Data</em>, vol. 7, no. 3, pp. 535&ndash;547, 2019.</li>
    <li>S. Robertson and H. Zaragoza, &ldquo;The probabilistic relevance framework: BM25 and beyond,&rdquo; <em>Found. Trends Inf. Retr.</em>, vol. 3, no. 4, pp. 333&ndash;389, 2009.</li>
    <li>G. V. Cormack, C. L. Clarke, and S. Buettcher, &ldquo;Reciprocal rank fusion outperforms condorcet and individual rank learning methods,&rdquo; in <em>Proc. ACM SIGIR</em>, pp. 758&ndash;759, 2009.</li>
    <li>R. Nogueira and K. Cho, &ldquo;Passage Re-ranking with BERT,&rdquo; <em>arXiv:1901.04085</em>, 2019.</li>
    <li>S. Es et al., &ldquo;RAGAS: Automated evaluation of retrieval augmented generation,&rdquo; <em>arXiv:2309.15217</em>, 2023.</li>
    <li>R. Smith, &ldquo;An overview of the Tesseract OCR engine,&rdquo; in <em>Proc. ICDAR</em>, vol. 2, pp. 629&ndash;633, 2007.</li>
</ol>

</body>
</html>
"""

html_path = os.path.abspath('adaptive_mmrag_journal_reference.html')
pdf_path = os.path.abspath('Adaptive_MMRAG_Research_Journal_Documentation.pdf')

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML written to: {html_path}")

cmd = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    f"--print-to-pdf={pdf_path}",
    html_path
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
if os.path.exists(pdf_path):
    print(f"SUCCESS: Generated PDF at {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path)} bytes")
else:
    print("FAILED: PDF not created")
    print(res.stderr)

