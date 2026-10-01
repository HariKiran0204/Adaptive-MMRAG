"""
adaptive_agents.py - Five specialized decision and processing agents for Adaptive-MMRAG.

Agents:
1. QueryAnalysisAgent - Analyzes intent, complexity, target modalities, reasoning requirements & query expansion.
2. RetrievalPlannerAgent - Formulates concrete adaptive retrieval strategy, Top-K budget, and reranker policy.
3. MultimodalRetrievalAgent - Executes retrieval with diversity-aware deduplication across FAISS, BM25, CLIP, and Whisper.
4. EvidenceVerificationAgent - Validates sufficiency and calculates composite confidence with keyword & entity coverage.
5. ResponseGenerationAgent - Formulates grounded, structured, hallucination-aware responses with exact citations.
"""

import re
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Generator
try:
    import ollama
except ImportError:
    ollama = None



from adaptive_config import (
    CONFIDENCE_THRESHOLDS,
    RE_RETRIEVAL_THRESHOLD,
    TOP_K_POLICIES,
    OLLAMA_MODEL,
    OLLAMA_VISION_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_KEEP_ALIVE,
    VALID_INTENTS,
    VALID_COMPLEXITIES,
    VALID_MODALITIES,
    VERIFICATION_FAITHFULNESS_THRESHOLD,
    ABSTENTION_MESSAGE,
    FULL_ABSTENTION_MESSAGE
)


# ============================================================================
# DATA STRUCTURES & TYPED RESULTS
# ============================================================================

@dataclass
class ClaimVerificationResult:
    """Output produced by PostGenerationVerifierAgent (Post-Generation Evidence Verification)."""
    total_claims: int
    supported_claims: int
    faithfulness_score: float     # 0.0 to 1.0 (fraction of claims supported)
    is_faithful: bool
    unsupported_claims: List[str]
    abstention_triggered: bool
    regeneration_triggered: bool
    final_answer: str


@dataclass
class QueryAnalysisResult:
    """Output produced by Agent 1 (Query Analysis Agent)."""
    intent: str
    complexity: str
    target_modalities: List[str]
    requires_reasoning: bool
    recommended_top_k: int
    is_cross_modal: bool
    is_overview: bool
    extracted_keywords: List[str] = field(default_factory=list)
    expanded_query: str = ""
    is_yes_no: bool = False          # True for polar Yes/No questions (e.g. "Is this suitable?")
    response_language: str = "en"   # ISO 639-1 code of requested response language


@dataclass
class RetrievalPlan:
    """Output produced by Agent 2 (Retrieval Planner Agent)."""
    strategy: str
    modalities: List[str]
    top_k: int
    use_reranker: bool
    overview_mode: bool
    iteration: int = 1
    reason: str = "initial_plan"


@dataclass
class VerificationResult:
    """Output produced by Agent 4 (Evidence Verification Agent)."""
    confidence_score: float  # Normalized 0.0 to 1.0
    confidence_level: str    # HIGH, MEDIUM, LOW
    is_sufficient: bool
    retrieved_count: int
    modality_breakdown: Dict[str, int]
    top_sources: List[str]
    verification_notes: str
    keyword_coverage: float = 1.0


@dataclass
class AdaptiveMetadata:
    """Comprehensive telemetry and audit trail for Adaptive-MMRAG queries."""
    intent: str
    complexity: str
    selected_modalities: List[str]
    retrieval_strategy: str
    top_k: int
    reranking: bool
    initial_evidence_score: float
    final_evidence_score: float
    retrieval_iterations: int
    re_retrieval: bool
    re_retrieval_reason: Optional[str] = None
    retrieval_latency_seconds: float = 0.0
    generation_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0
    answer_length_chars: int = 0
    top_sources: str = ""


# ============================================================================
# AGENT 1 — QUERY ANALYSIS AGENT (Enhanced with Entity & Synonym Expansion)
# ============================================================================

class QueryAnalysisAgent:
    """
    Agent 1: Performs deterministic & heuristic query understanding.
    Analyzes intent, complexity, modality targets, and reasoning needs in < 2ms
    without stalling CPU on heavy LLM classification calls.
    """
    
    def __init__(self):
        # Intent patterns
        self.overview_keywords = {
            "summarize", "summary", "overview", "what is present", "what is in", 
            "what does the document contain", "topics covered", "about this document",
            "tell me about this file", "explain the document", "explain the file",
            "describe the document", "general idea", "all details", "from starting to ending",
            # Image overview phrases — "what is the image about?", "describe this image", etc.
            "what is the image about", "what is this image", "what is this image about",
            "tell me about this image", "tell me about this photo", "tell me about this picture",
            "what does this image show", "what does the image show", "what does this picture show",
            "describe this image", "describe this photo", "describe the image",
            "what is in this image", "what is in the image", "what is in this photo",
            "about this image", "about this photo", "about this picture",
            "explain this image", "explain the image", "explain this photo",
        }
        self.comparison_keywords = {"compare", "difference", "versus", "vs", "similarities", "distinguish", "contrast"}
        self.reasoning_keywords = {"why", "how", "explain", "analyze", "evaluate", "step by step", "derivation", "cause", "effect"}
        
        # Modality patterns
        self.image_keywords = {
            "image", "photo", "picture", "figure", "diagram", "chart", "graph", 
            "screenshot", "visual", "look like", "barcode", "pan card", "id card", 
            "logo", "signature", "stamp", "certificate", "drawn", "layout"
        }
        self.audio_keywords = {
            "audio", "voice", "recording", "speaker", "spoke", "said", 
            "listen", "speech", "transcript", "conversation", "podcast", "talk"
        }
        self.table_keywords = {
            "table", "row", "column", "data sheet", "tabular", "matrix", "values", "statistics"
        }

        # Yes/No question detection — these starters make a polar question
        # e.g. "Is this suitable?", "Does this match?", "Can he apply?"
        self.yes_no_starters = {
            "is", "are", "does", "do", "can", "will", "should", "was", "were",
            "has", "have", "would", "could", "did", "might"
        }
        # Suitability / eligibility words that strengthen yes_no classification
        self.yes_no_suitability_words = {
            "suitable", "right", "good", "appropriate", "qualified", "eligible",
            "fit", "match", "relevant", "applicable", "valid", "correct", "okay",
            "ok", "perfect", "ideal", "enough", "sufficient", "recommended",
            "compatible", "capable", "ready", "useful"
        }

        # Synonym expansion map for high-precision retrieval
        self.synonym_map = {
            "pan": ["pan card", "permanent account number", "pan no"],
            "income": ["income certificate", "annual income", "salary"],
            "application": ["application number", "application no", "app no", "acknowledgement"],
            "dob": ["date of birth", "birth date", "born on"],
            "father": ["father name", "father's name", "parent", "guardian"],
            "address": ["residence", "locality", "door no", "street", "district", "pin"],
            "validity": ["valid up to", "validity period", "expiry date", "valid for", "years validity"],
            "revenue": ["total revenue", "sales", "turnover", "financial", "profit"]
        }

    def analyze(self, query: str, available_modalities: Optional[List[str]] = None) -> QueryAnalysisResult:
        """Analyze user query and return structured analysis result."""
        q = query.strip().lower()
        
        # 1. Intent Detection
        if any(kw in q for kw in self.overview_keywords):
            intent = "document_overview"
            is_overview = True
        elif any(kw in q for kw in self.comparison_keywords):
            intent = "comparison"
            is_overview = False
        elif any(kw in q for kw in self.reasoning_keywords):
            intent = "reasoning"
            is_overview = False
        elif any(kw in q for kw in self.image_keywords) and any(kw in q for kw in self.audio_keywords):
            intent = "cross_modal"
            is_overview = False
        elif any(kw in q for kw in self.image_keywords):
            intent = "visual_question"
            is_overview = False
        elif any(kw in q for kw in self.audio_keywords):
            intent = "audio_question"
            is_overview = False
        elif len(q.split()) <= 4 and ("what is" in q or "who is" in q or "number" in q or "date" in q or "name" in q):
            intent = "factual"
            is_overview = False
        else:
            intent = "factual"
            is_overview = False
            
        # 2. Modality Selection
        explicit_modalities = []
        if any(kw in q for kw in self.image_keywords):
            explicit_modalities.append("IMAGE")
        if any(kw in q for kw in self.audio_keywords):
            explicit_modalities.append("AUDIO")
        if any(kw in q for kw in self.table_keywords):
            explicit_modalities.append("TABLE")

        is_cross_modal = len(explicit_modalities) > 1

        if not explicit_modalities:
            target_modalities = ["TEXT"]
        else:
            target_modalities = list(explicit_modalities)
            if "TEXT" not in target_modalities:
                target_modalities.append("TEXT")

        # 3. Complexity Estimation
        words = q.split()
        word_count = len(words)
        
        if is_overview or intent in ("comparison", "reasoning") or word_count > 15:
            complexity = "HIGH"
            requires_reasoning = True
            top_k = TOP_K_POLICIES["OVERVIEW"] if is_overview else TOP_K_POLICIES["HIGH"]
        elif word_count >= 7 or is_cross_modal:
            complexity = "MEDIUM"
            requires_reasoning = intent in ("reasoning", "comparison")
            top_k = TOP_K_POLICIES["MEDIUM"]
        else:
            complexity = "LOW"
            requires_reasoning = False
            top_k = TOP_K_POLICIES["LOW"]

        # 4. Keyword Extraction & Synonym Enrichment
        stopwords = {
            "what", "is", "the", "in", "that", "this", "can", "you", "tell", "me", 
            "a", "an", "of", "and", "or", "to", "for", "it", "from", "on", "with", "about"
        }
        keywords = [w for w in re.findall(r'\b\w+\b', q) if w not in stopwords and len(w) > 1]

        # Build expanded query string
        expanded_terms = list(keywords)
        for word in keywords:
            if word in self.synonym_map:
                expanded_terms.extend(self.synonym_map[word])
        
        # Build expanded query string
        expanded_query = " ".join(list(dict.fromkeys(expanded_terms)))

        # 5. Yes/No Question Detection
        # Detect polar questions: "Is this suitable?", "Does he qualify?", "Can this be used?"
        words_raw = query.strip().split()
        first_word = words_raw[0].lower().rstrip('?,') if words_raw else ''
        has_suitability_word = any(sw in q for sw in self.yes_no_suitability_words)
        is_yes_no = (
            first_word in self.yes_no_starters
            and (has_suitability_word or len(words_raw) <= 12)
            and not is_overview
        )

        return QueryAnalysisResult(
            intent=intent,
            complexity=complexity,
            target_modalities=target_modalities,
            requires_reasoning=requires_reasoning,
            recommended_top_k=top_k,
            is_cross_modal=is_cross_modal,
            is_overview=is_overview,
            extracted_keywords=keywords,
            expanded_query=expanded_query,
            is_yes_no=is_yes_no,
        )



# ============================================================================
# AGENT 2 — RETRIEVAL PLANNER AGENT
# ============================================================================

class RetrievalPlannerAgent:
    """
    Agent 2: Converts query analysis into an optimized, concrete retrieval plan.
    Determines search strategy, Top-K budget, reranking activation, and overview mode.
    """
    
    def plan(
        self, 
        analysis: QueryAnalysisResult, 
        total_indexed_chunks: int, 
        iteration: int = 1,
        re_retrieval_context: Optional[Dict[str, Any]] = None
    ) -> RetrievalPlan:
        """Formulate retrieval plan based on query analysis and iteration state."""
        
        # Base settings
        top_k = analysis.recommended_top_k
        modalities = list(analysis.target_modalities)
        overview_mode = analysis.is_overview
        
        # Strategy selection
        if overview_mode:
            strategy = "global_document_overview"
            use_reranker = False
            top_k = min(20, max(12, total_indexed_chunks))
        elif analysis.complexity == "LOW":
            strategy = "hybrid_semantic_bm25_fast"
            use_reranker = False  # Skip expensive cross-encoder for simple lookups to save 200ms
        elif analysis.complexity == "HIGH" or analysis.is_cross_modal:
            strategy = "hybrid_semantic_bm25_rrf_cross_encoder"
            use_reranker = True
        else:
            strategy = "hybrid_semantic_bm25_rrf_cross_encoder"
            use_reranker = True

        # Handle Re-retrieval Adaptation (Iteration 2)
        reason = "initial_plan"
        if iteration > 1 and re_retrieval_context:
            reason = "insufficient_evidence_expansion"
            # 1. Expand Top-K
            top_k = int(math.ceil(top_k * TOP_K_POLICIES["RE_RETRIEVE_EXPANSION"]))
            # 2. Force enable reranking for maximum precision
            use_reranker = True
            # 3. Broaden modalities if restricted
            if "TEXT" not in modalities:
                modalities.append("TEXT")
            if "IMAGE" not in modalities and re_retrieval_context.get("has_images", False):
                modalities.append("IMAGE")
            strategy = "adaptive_broadened_hybrid_rerank"

        return RetrievalPlan(
            strategy=strategy,
            modalities=modalities,
            top_k=min(top_k, max(1, total_indexed_chunks)),
            use_reranker=use_reranker,
            overview_mode=overview_mode,
            iteration=iteration,
            reason=reason
        )


# ============================================================================
# AGENT 3 — MULTIMODAL RETRIEVAL AGENT (Enhanced with Diversity Deduplication)
# ============================================================================

class MultimodalRetrievalAgent:
    """
    Agent 3: Executes the retrieval plan cleanly using the existing search engines.
    Includes diversity-aware deduplication across FAISS, BM25, CLIP, and Whisper.
    """
    
    def execute(self, plan: RetrievalPlan, query: str, chatbot_instance, analysis: Optional[QueryAnalysisResult] = None) -> List[Dict[str, Any]]:
        """Execute retrieval plan on the vector store and hybrid search engine."""
        if chatbot_instance is None or not hasattr(chatbot_instance, 'search_engine'):
            return []

        # 1. Overview Strategy
        if plan.overview_mode and chatbot_instance.metadata:
            results = []
            for i, meta in enumerate(chatbot_instance.metadata):
                orig = meta.get('original_metadata', {})
                if orig.get('is_overview', False) or meta.get('is_overview', False):
                    results.append({'score': 1.0, 'rank': 1, 'metadata': meta})
            
            remaining_budget = max(0, plan.top_k - len(results))
            if remaining_budget > 0:
                num_samples = min(remaining_budget, len(chatbot_instance.metadata))
                indices = [int(i * (len(chatbot_instance.metadata) / num_samples)) for i in range(num_samples)]
                for i, idx in enumerate(indices):
                    if chatbot_instance.metadata[idx] not in [r['metadata'] for r in results]:
                        results.append({
                            'score': 1.0,
                            'rank': len(results) + 1,
                            'metadata': chatbot_instance.metadata[idx]
                        })
            return results or [{'score': 1.0, 'rank': 1, 'metadata': chatbot_instance.metadata[0]}]

        # 2. Hybrid Search Strategy (with Expanded Query if available)
        search_query = query
        if analysis and analysis.expanded_query and len(analysis.expanded_query) > len(query):
            search_query = f"{query} {analysis.expanded_query}"

        raw_results = chatbot_instance.search_by_text(
            search_query, 
            k=plan.top_k + 4,  # Retrieve extra candidates for diversity filtering
            use_reranker=plan.use_reranker
        )

        # Filter by modality if plan specifies strict modalities
        if plan.modalities and "TEXT" not in plan.modalities:
            filtered = [r for r in raw_results if r.get('metadata', {}).get('type', '').upper() in plan.modalities]
            raw_results = filtered or raw_results

        # 3. Diversity Deduplication: Filter out near-identical duplicate text chunks
        unique_results = []
        seen_snippets = set()

        for res in raw_results:
            content = res.get('metadata', {}).get('content', '').strip()
            # Generate 40-char fingerprint
            fingerprint = content[:40].lower() if len(content) >= 40 else content.lower()
            if fingerprint and fingerprint not in seen_snippets:
                seen_snippets.add(fingerprint)
                unique_results.append(res)
            elif not fingerprint:
                unique_results.append(res)

            if len(unique_results) >= plan.top_k:
                break

        return unique_results or raw_results[:plan.top_k]


# ============================================================================
# AGENT 4 — EVIDENCE VERIFICATION AGENT (Enhanced with Entity Coverage)
# ============================================================================

class EvidenceVerificationAgent:
    """
    Agent 4: Validates sufficiency of retrieved context before generation.
    Computes a composite confidence score combining relevance, top-3 mean, 
    content richness, and keyword/entity coverage.
    """
    
    def verify(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        plan: RetrievalPlan,
        analysis: Optional[QueryAnalysisResult] = None
    ) -> VerificationResult:
        """Evaluate evidence quality and return VerificationResult."""
        if not results:
            return VerificationResult(
                confidence_score=0.0,
                confidence_level="LOW",
                is_sufficient=False,
                retrieved_count=0,
                modality_breakdown={},
                top_sources=[],
                verification_notes="No evidence chunks retrieved.",
                keyword_coverage=0.0
            )

        # Normalize scores to [0, 1] range
        normalized_scores = []
        for r in results:
            s = r.get('score', 0.5)
            if s > 1.0 and s < 100.0:  # Cross-encoder logit
                prob = 1.0 / (1.0 + math.exp(-s))
            elif s > 1.0:
                prob = 1.0
            else:
                prob = s
            normalized_scores.append(prob)

        top_score = max(normalized_scores) if normalized_scores else 0.0
        mean_top3 = float(sum(normalized_scores[:3]) / min(3, len(normalized_scores))) if normalized_scores else 0.0
        
        # Check chunk content richness
        total_chars = sum(len(r.get('metadata', {}).get('content', '')) for r in results)
        content_richness = min(1.0, total_chars / 1500.0)

        # Modality breakdown & Corpus assembly
        modalities_found = {}
        corpus_list = []
        for r in results:
            m = r.get('metadata', {}).get('type', 'text').lower()
            modalities_found[m] = modalities_found.get(m, 0) + 1
            corpus_list.append(r.get('metadata', {}).get('content', '').lower())
        
        evidence_corpus = " ".join(corpus_list)

        # Keyword & Entity Coverage Check
        keyword_coverage = 1.0
        if analysis and analysis.extracted_keywords:
            matched = sum(1 for kw in analysis.extracted_keywords if kw.lower() in evidence_corpus)
            keyword_coverage = matched / float(len(analysis.extracted_keywords))

        # Composite Confidence Formula:
        # 35% top score + 25% mean top-3 + 20% keyword coverage + 10% content richness + 10% chunk count
        chunk_count_factor = min(1.0, len(results) / float(plan.top_k))
        confidence = (0.35 * top_score) + (0.25 * mean_top3) + (0.20 * keyword_coverage) + (0.10 * content_richness) + (0.10 * chunk_count_factor)
        confidence = round(min(1.0, max(0.0, confidence)), 3)

        # Determine level
        if confidence >= CONFIDENCE_THRESHOLDS["HIGH"]:
            level = "HIGH"
            is_sufficient = True
        elif confidence >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
            level = "MEDIUM"
            is_sufficient = True
        else:
            level = "LOW"
            is_sufficient = False

        top_sources = list(set([r.get('metadata', {}).get('source', 'Unknown') for r in results[:5]]))
        notes = f"Evidence score {confidence:.2f} ({level}). Keyword coverage: {keyword_coverage:.0%}. {len(results)} chunks supporting query."

        return VerificationResult(
            confidence_score=confidence,
            confidence_level=level,
            is_sufficient=is_sufficient,
            retrieved_count=len(results),
            modality_breakdown=modalities_found,
            top_sources=top_sources,
            verification_notes=notes,
            keyword_coverage=keyword_coverage
        )


# ============================================================================
# AGENT 5 — RESPONSE GENERATION AGENT (Enhanced with Structured Grounding)
# ============================================================================

class ResponseGenerationAgent:
    """
    Agent 5: Generates grounded, hallucination-aware responses from verified evidence.
    Supports both token streaming and complete generation with citation guardrails.
    """
    
    def build_grounded_prompt(
        self, 
        query: str, 
        context: Dict[str, Any], 
        verification: VerificationResult,
        analysis: QueryAnalysisResult,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Formulate hallucination-aware prompt with evidence confidence badges, conversation history, and structured rules."""
        evidence_sections = []

        # ----------------------------------------------------------------
        # Adaptive chunk scaling based on query type
        # Yes/No and simple factual queries get fewer chunks so the LLM
        # doesn't see the entire document and decide to summarize it.
        # ----------------------------------------------------------------
        if analysis.is_yes_no:
            max_text_chunks = 3
            max_chunk_chars = 1200
        elif analysis.is_overview or analysis.complexity == "HIGH":
            max_text_chunks = 8
            max_chunk_chars = 1800
        else:
            max_text_chunks = 4      # was 6 — reduced to keep answers focused
            max_chunk_chars = 1400

        # Text chunks
        for item in context.get('text', [])[:max_text_chunks]:
            source_tag = f"Source: {item.get('source', 'Doc')}"
            if item.get('page_number'):
                source_tag += f" (Page {item['page_number']})"
            evidence_sections.append(f"{source_tag}\n{item.get('content', '')[:max_chunk_chars]}")

        # Table chunks
        for item in context.get('tables', [])[:3]:
            source_tag = f"Source: {item.get('source', 'Doc')} [Table]"
            evidence_sections.append(f"{source_tag}\n{item.get('content', '')[:1400]}")

        # Image OCR & description chunks
        for item in context.get('images', [])[:3]:
            source_tag = f"Source: {item.get('source', 'Doc')} [Image/Document]"
            evidence_sections.append(f"{source_tag}\n{item.get('content', '')[:3500]}")

        # Audio transcript chunks
        max_audio_chunks = 8 if (analysis.is_overview or analysis.complexity == "HIGH") else 5
        for item in context.get('audio', [])[:max_audio_chunks]:
            source_tag = f"Source: {item.get('source', 'Audio')} [Transcript]"
            evidence_sections.append(f"{source_tag}\n{item.get('content', '')[:max_chunk_chars]}")

        evidence_text = "\n\n".join(evidence_sections)

        # ----------------------------------------------------------------
        # Grounding instruction — image queries, Yes/No, LOW confidence, overview, factual
        # ----------------------------------------------------------------
        has_image_evidence = (
            len(context.get('images', [])) > 0
            or any(item.get('source', '').lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')) for item in context.get('images', []) + context.get('text', []))
        )
        is_image_query = (
            has_image_evidence and (
                "IMAGE" in analysis.target_modalities
                or analysis.intent in ("visual_question", "cross_modal")
                or any(w in query.lower() for w in ("image", "photo", "picture", "diagram", "show", "what is this", "what is the image"))
            )
        )

        if is_image_query:
            grounding_rule = (
                "Instructions for Image Understanding:\n"
                "1. State clearly and directly what the image depicts/shows in the very first sentence (e.g. '**This image is about an Apple.**').\n"
                "2. Provide relevant visual details, subject name, and file information from the evidence.\n"
                "3. Keep the answer direct, natural, and concise (under 80 words). Do NOT invent document sections, business metrics, or unrelated text."
            )
        elif analysis.is_yes_no:
            # Polar Yes/No question — direct, minimal, no document dump
            grounding_rule = (
                "Instructions for Yes/No Questions (STRICT):\n"
                "1. First line: write ONLY '**YES**' or '**NO**' in bold — nothing else on that line.\n"
                "2. Then write 2-3 short bullet points explaining WHY, using only the evidence below.\n"
                "3. Total response MUST be under 80 words. Do NOT summarize the whole document.\n"
                "4. Do NOT list all skills/sections — only the most relevant facts to answer the question.\n"
                "5. End with [Source: filename]."
            )
        elif verification.confidence_level == "LOW":
            grounding_rule = (
                "NOTE: Retrieved evidence is partial. Answer carefully based strictly on "
                "the document evidence below. Extract all exact names, numbers, dates, and details visible."
            )
        elif analysis.is_overview:
            grounding_rule = (
                "Instructions for Comprehensive Document Overview:\n"
                "Provide an exhaustive, thorough overview of the entire document based on the evidence below. "
                "Structure your response with clear headings and bullet points covering all main sections, key findings, "
                "quantitative metrics, requirements, and conclusions. Do not omit important sections."
            )
        else:
            grounding_rule = (
                "Instructions for Crisp, Direct Responses:\n"
                "1. Direct & Factual: Answer the question directly in the first sentence in bold.\n"
                "2. Key Details: Follow with 3-5 concise, structured bullet points highlighting the exact names, metrics, skills, or facts from the document.\n"
                "3. Crisp & Focused: Be crisp, direct, and concise (target: 100-180 words). Avoid conversational filler or long essays.\n"
                "4. General Knowledge Fallback: If the indexed documents don't have relevant information for the question, answer using your general knowledge. Do NOT just say 'not found' — provide a helpful, accurate best-effort answer.\n"
                "5. Citations: Append exact source citations at the end (e.g. [Source: filename.pdf]) only when using document evidence."
            )

        # Language instruction (only added when user requests non-English response)
        lang_code = getattr(analysis, 'response_language', 'en')
        lang_names = {
            'te': 'Telugu', 'hi': 'Hindi', 'ta': 'Tamil', 'kn': 'Kannada',
            'ml': 'Malayalam', 'mr': 'Marathi', 'bn': 'Bengali',
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ar': 'Arabic'
        }
        lang_instruction = ""
        if lang_code and lang_code != 'en' and lang_code in lang_names:
            lang_instruction = f"\n\nIMPORTANT: Respond entirely in {lang_names[lang_code]}."

        # Build conversation history context (Bug 10 — multi-turn memory)
        history_section = ""
        history = conversation_history or []
        if history:
            history_lines = []
            for turn in history[-6:]:  # Keep last 6 turns max to control prompt size
                role = turn.get('role', 'user')
                content = str(turn.get('content', '')).strip()
                if content:
                    history_lines.append(f"{role.upper()}: {content[:500]}")  # Truncate long turns
            if history_lines:
                history_section = "CONVERSATION HISTORY (most recent turns):\n" + "\n".join(history_lines) + "\n\n"

        # Universal typo/grammar correction notice (Bug 8)
        typo_notice = (
            "NOTE: If the user's question contains typos, grammar errors, or unclear wording, "
            "silently correct the intent and answer the corrected version without mentioning the error.\n\n"
        )

        return (
            f"You are Adaptive-MMRAG, an expert offline multimodal AI assistant.\n"
            f"{typo_notice}"
            f"{grounding_rule}\n\n"
            f"{history_section}"
            f"DOCUMENT EVIDENCE (Confidence: {verification.confidence_level}):\n"
            f"{evidence_text}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:{lang_instruction}"
        )

    def generate_stream(
        self, 
        prompt: str, 
        image_paths: Optional[List[str]] = None,
        use_vision_model: bool = False,
        selected_model: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream generated response tokens from local Ollama model using routed model."""
        target_model = selected_model or OLLAMA_MODEL
        try:
            if use_vision_model and image_paths:
                stream = ollama.chat(
                    model=OLLAMA_VISION_MODEL,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': image_paths[:3]
                    }],
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=OLLAMA_KEEP_ALIVE
                )
                for chunk in stream:
                    yield chunk['message']['content']
            else:
                stream = ollama.generate(
                    model=target_model,
                    prompt=prompt,
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=OLLAMA_KEEP_ALIVE
                )
                for chunk in stream:
                    yield chunk.get('response', '')
        except Exception as e:
            yield f"\n[Generation Error ({target_model}): {e}. Ensure Ollama is running: ollama serve]"


# ============================================================================
# AGENT 6 — POST-GENERATION EVIDENCE VERIFIER & ABSTENTION AGENT
# ============================================================================

class PostGenerationVerifierAgent:
    """
    Agent 6: Verifies factual claims in generated answers against retrieved evidence chunks.
    Enforces hallucination mitigation, claim support scoring, and formal abstention.
    
    Pipeline:
    1. Extract discrete factual claims/sentences from generated answer.
    2. Compute support score per claim against evidence corpus (lexical + entity matching + CrossEncoder).
    3. Calculate aggregate faithfulness score (fraction of supported claims).
    4. Enforce abstention or partial warning if faithfulness < VERIFICATION_FAITHFULNESS_THRESHOLD.
    """

    def __init__(self, threshold: float = VERIFICATION_FAITHFULNESS_THRESHOLD):
        self.threshold = threshold
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "and", "or", "that", "this", "it", "as", "be",
            "have", "has", "had", "will", "would", "can", "could", "should", "not", "but"
        }

    def extract_claims(self, answer: str) -> List[str]:
        """Extract discrete factual sentences/claims from the answer."""
        if not answer or not answer.strip():
            return []

        # Remove citation blocks like [Source: ...] or markdown headers
        cleaned = re.sub(r'\[Source:.*?\]', '', answer, flags=re.IGNORECASE)
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
        
        # Split into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', cleaned)
        claims = []

        for s in raw_sentences:
            s_clean = s.strip()
            # Strip markdown bullets / numbering
            s_clean = re.sub(r'^[\*\-\#\d\.\s\:]+', '', s_clean).strip()
            # Filter out non-factual or trivial short strings
            if len(s_clean) >= 15 and not s_clean.lower().startswith(("here is", "in summary", "overall,", "according to")):
                claims.append(s_clean)

        return claims or ([cleaned.strip()] if len(cleaned.strip()) >= 10 else [])

    def verify_post_generation(
        self,
        query: str,
        answer: str,
        evidence_chunks: List[Dict[str, Any]],
        cross_encoder: Optional[Any] = None
    ) -> ClaimVerificationResult:
        """
        Evaluate factual grounding of answer against evidence chunks.
        
        Returns ClaimVerificationResult with support metrics and verified answer.
        """
        if not answer or not answer.strip():
            return ClaimVerificationResult(
                total_claims=0,
                supported_claims=0,
                faithfulness_score=0.0,
                is_faithful=False,
                unsupported_claims=[],
                abstention_triggered=True,
                regeneration_triggered=False,
                final_answer=FULL_ABSTENTION_MESSAGE
            )

        # Build evidence corpus — handles both text chunks ('content') and
        # image chunks ('searchable_text', 'content_description', 'ocr_text', etc.)
        corpus_parts = []
        for chunk in evidence_chunks:
            meta = chunk.get('metadata', {})
            # Text/document chunks use 'content'
            piece = meta.get('content', '')
            if not piece:
                # Image chunks — aggregate all available textual signals
                piece = ' '.join(filter(None, [
                    meta.get('searchable_text', ''),
                    meta.get('content_description', ''),
                    meta.get('ocr_text', ''),
                    meta.get('user_description', ''),
                    ' '.join(meta.get('keywords', [])) if isinstance(meta.get('keywords'), list) else str(meta.get('keywords', '')),
                    meta.get('figure_title', ''),
                ]))
            if piece.strip():
                corpus_parts.append(piece)
        
        evidence_corpus = " ".join(corpus_parts).lower()
        if not evidence_corpus.strip():
            # Truly no text evidence at all — still abstain, but this should be
            # very rare now that image text fields are included above.
            return ClaimVerificationResult(
                total_claims=1,
                supported_claims=0,
                faithfulness_score=0.0,
                is_faithful=False,
                unsupported_claims=[answer[:100]],
                abstention_triggered=True,
                regeneration_triggered=False,
                final_answer=FULL_ABSTENTION_MESSAGE
            )

        claims = self.extract_claims(answer)
        if not claims:
            return ClaimVerificationResult(
                total_claims=0,
                supported_claims=0,
                faithfulness_score=1.0,
                is_faithful=True,
                unsupported_claims=[],
                abstention_triggered=False,
                regeneration_triggered=False,
                final_answer=answer
            )

        supported_count = 0
        unsupported_claims = []

        # Batch predict CrossEncoder scores in a single pass to save seconds
        ce_scores = {}
        if cross_encoder is not None and len(evidence_corpus) > 20 and claims:
            try:
                summary_context = evidence_corpus[:600]
                pairs = [[c, summary_context] for c in claims]
                batch_scores = cross_encoder.predict(pairs, batch_size=16)
                for c, sc in zip(claims, batch_scores):
                    ce_scores[c] = float(sc) > -2.0
            except Exception:
                pass

        for claim in claims:
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', claim) if w.lower() not in self.stopwords]
            if not words:
                supported_count += 1
                continue

            matched_words = sum(1 for w in words if w in evidence_corpus)
            overlap_ratio = matched_words / float(len(words))

            numbers_in_claim = re.findall(r'\b\d+(?:[\.,\-]\d+)?\b', claim)
            numbers_matched = all(num in evidence_corpus for num in numbers_in_claim) if numbers_in_claim else True

            ce_supported = ce_scores.get(claim, True)

            if overlap_ratio >= 0.35 and numbers_matched and ce_supported:
                supported_count += 1
            else:
                unsupported_claims.append(claim)

        faithfulness = round(supported_count / float(len(claims)), 3)
        is_faithful = faithfulness >= self.threshold

        abstention_triggered = False
        final_answer = answer

        if not is_faithful:
            abstention_triggered = True
            if faithfulness < 0.25:
                # Severe hallucination / zero evidence -> Full abstention
                final_answer = FULL_ABSTENTION_MESSAGE
            else:
                # Partial support -> Prefix warning banner
                warning_prefix = f"⚠️ [{ABSTENTION_MESSAGE}]\n\n"
                final_answer = warning_prefix + answer

        return ClaimVerificationResult(
            total_claims=len(claims),
            supported_claims=supported_count,
            faithfulness_score=faithfulness,
            is_faithful=is_faithful,
            unsupported_claims=unsupported_claims,
            abstention_triggered=abstention_triggered,
            regeneration_triggered=False,
            final_answer=final_answer
        )

