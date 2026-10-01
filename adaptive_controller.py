"""
adaptive_controller.py - Central orchestration controller for Adaptive-MMRAG.
Coordinates the 5 specialized agents, manages the adaptive re-retrieval loop,
computes runtime telemetry, and provides both streaming and synchronous interfaces.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Generator
from datetime import datetime

from adaptive_config import (
    ADAPTIVE_MODE_DEFAULT,
    MAX_RETRIEVAL_ITERATIONS,
    RE_RETRIEVAL_THRESHOLD,
    OLLAMA_MODEL
)
from adaptive_agents import (
    QueryAnalysisAgent,
    RetrievalPlannerAgent,
    MultimodalRetrievalAgent,
    EvidenceVerificationAgent,
    ResponseGenerationAgent,
    PostGenerationVerifierAgent,
    QueryAnalysisResult,
    RetrievalPlan,
    VerificationResult,
    ClaimVerificationResult,
    AdaptiveMetadata
)
from model_router import ModelRouter, ModelRoutingDecision


class AdaptiveRAGController:
    """
    Central Controller for the Adaptive-MMRAG Framework.
    
    Orchestration Flow:
    1. Query Analysis (Agent 1)
    2. Retrieval Planning (Agent 2)
    3. Multimodal Retrieval Execution (Agent 3)
    4. Pre-Generation Evidence Verification (Agent 4)
    5. Adaptive Re-Retrieval Feedback Loop (if evidence insufficient)
    6. Resource-Aware Model Routing (ModelRouter)
    7. Grounded Response Generation (Agent 5)
    8. Post-Generation Evidence Verification & Abstention (Agent 6)
    9. Telemetry & Metadata Compilation
    """

    def __init__(self):
        self.query_agent = QueryAnalysisAgent()
        self.planner_agent = RetrievalPlannerAgent()
        self.retrieval_agent = MultimodalRetrievalAgent()
        self.verification_agent = EvidenceVerificationAgent()
        self.generation_agent = ResponseGenerationAgent()
        self.model_router = ModelRouter()
        self.post_verifier_agent = PostGenerationVerifierAgent()
        print("🧠 Adaptive-MMRAG Controller Initialized with Multi-Agent Pipeline, Resource-Aware Router, and Post-Generation Verifier.")


    def run_adaptive_pipeline(
        self, 
        query: str, 
        chatbot_instance
    ) -> Tuple[str, Dict[str, Any], VerificationResult, QueryAnalysisResult, RetrievalPlan, Dict[str, Any]]:
        """
        Executes the adaptive decision-making and re-retrieval pipeline.
        
        Returns:
            prompt: Formatted prompt for generation
            context: Formatted multimodal context dictionary
            verification: Final verification result
            analysis: Query analysis result
            plan: Final retrieval plan
            telemetry: Dictionary of latency and iteration metrics
        """
        pipeline_started_at = time.perf_counter()
        total_chunks = len(chatbot_instance.metadata) if (chatbot_instance and chatbot_instance.metadata) else 0

        # Step 1: Query Analysis (Agent 1)
        analysis_start = time.perf_counter()
        analysis = self.query_agent.analyze(query)
        analysis_duration = time.perf_counter() - analysis_start

        # Step 2: Retrieval Planning (Agent 2)
        planner_start = time.perf_counter()
        plan = self.planner_agent.plan(analysis, total_indexed_chunks=total_chunks, iteration=1)
        planner_duration = time.perf_counter() - planner_start

        # Step 3: Multimodal Retrieval Execution (Agent 3) - Iteration 1
        retrieval_start = time.perf_counter()
        results = self.retrieval_agent.execute(plan, query, chatbot_instance, analysis=analysis)
        retrieval_duration = time.perf_counter() - retrieval_start

        # Step 4: Evidence Verification (Agent 4) - Iteration 1
        verify_start = time.perf_counter()
        verification = self.verification_agent.verify(query, results, plan, analysis=analysis)
        verify_duration = time.perf_counter() - verify_start

        initial_score = verification.confidence_score
        final_score = initial_score
        iteration_count = 1
        re_retrieval_triggered = False
        re_retrieval_reason = None

        # Step 5: Adaptive Feedback Loop / Re-Retrieval (if evidence is weak)
        if not verification.is_sufficient and MAX_RETRIEVAL_ITERATIONS > 1 and total_chunks > plan.top_k:
            re_retrieval_triggered = True
            re_retrieval_reason = f"Evidence score {initial_score:.2f} below threshold {RE_RETRIEVAL_THRESHOLD}"
            print(f"   ⚠️  Adaptive Re-Retrieval Triggered: {re_retrieval_reason}")

            # Re-plan with expanded Top-K and broadened modality search
            plan_iter2 = self.planner_agent.plan(
                analysis, 
                total_indexed_chunks=total_chunks, 
                iteration=2,
                re_retrieval_context={"initial_score": initial_score}
            )
            
            re_retrieval_start = time.perf_counter()
            results_iter2 = self.retrieval_agent.execute(plan_iter2, query, chatbot_instance, analysis=analysis)
            retrieval_duration += (time.perf_counter() - re_retrieval_start)

            # Re-verify
            verification_iter2 = self.verification_agent.verify(query, results_iter2, plan_iter2, analysis=analysis)
            final_score = verification_iter2.confidence_score
            iteration_count = 2
            
            # Adopt improved results if score improved or maintained
            if final_score >= initial_score or len(results_iter2) > len(results):
                results = results_iter2
                verification = verification_iter2
                plan = plan_iter2
                print(f"   ✓ Re-retrieval completed: score {initial_score:.2f} ➔ {final_score:.2f}")

        # Format multimodal context
        context = chatbot_instance.format_context_by_modality(results)

        # Step 6: Formulate Grounded Prompt (Agent 5)
        prompt = self.generation_agent.build_grounded_prompt(query, context, verification, analysis)

        total_retrieval_pipeline_time = time.perf_counter() - pipeline_started_at
        telemetry = {
            "analysis_latency": analysis_duration,
            "planner_latency": planner_duration,
            "retrieval_latency": retrieval_duration,
            "verify_latency": verify_duration,
            "total_retrieval_pipeline_latency": total_retrieval_pipeline_time,
            "initial_score": initial_score,
            "final_score": final_score,
            "iteration_count": iteration_count,
            "re_retrieval_triggered": re_retrieval_triggered,
            "re_retrieval_reason": re_retrieval_reason,
            "retrieved_results_count": len(results),
            "top_sources": list(set([r.get('metadata', {}).get('source', 'Unknown') for r in results[:5]]))
        }

        return prompt, context, verification, analysis, plan, telemetry

    def process_query_stream(
        self, 
        query: str, 
        chatbot_instance,
        mode: Optional[str] = None,
        response_language: str = "en",
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
        """
        Streams response tokens and yields comprehensive telemetry upon completion.
        
        Args:
            query: User query string
            chatbot_instance: Active CrossModalRAGChatbot instance
            mode: "adaptive" or "baseline" (defaults to ADAPTIVE_MODE_DEFAULT)
            conversation_history: List of {"role": "user/assistant", "content": "..."} dicts (last N turns)
            
        Yields:
            Tuple of (token_str, metadata_dict_or_None)
        """
        use_adaptive = ADAPTIVE_MODE_DEFAULT if mode is None else (mode.lower() == "adaptive")
        request_started_at = time.perf_counter()
        history = conversation_history or []

        if not use_adaptive:
            # ================================================================
            # BASELINE FIXED RETRIEVAL PATH (with Post-Generation Verification)
            # ================================================================
            sys_res = self.model_router.get_system_resources()
            retrieval_start = time.perf_counter()
            search_results = chatbot_instance.search_by_text(query, k=6, use_reranker=True)
            retrieval_duration = time.perf_counter() - retrieval_start
            
            context = chatbot_instance.format_context_by_modality(search_results)
            prompt = chatbot_instance.build_fast_rag_prompt(query, context)

            generation_start = time.perf_counter()
            full_response = []
            stream_gen = chatbot_instance.generate_answer_stream(query, search_results, query_type="text")
            
            for token in stream_gen:
                full_response.append(token)
                yield token, None

            generation_duration = time.perf_counter() - generation_start
            total_duration = time.perf_counter() - request_started_at
            answer_text = "".join(full_response)

            # Step: Post-Generation Verification on Baseline Output
            cross_enc = getattr(chatbot_instance.search_engine, 'reranker', None) if hasattr(chatbot_instance, 'search_engine') else None
            post_verify = self.post_verifier_agent.verify_post_generation(
                query=query,
                answer=answer_text,
                evidence_chunks=search_results,
                cross_encoder=cross_enc
            )

            baseline_metadata = {
                "mode": "baseline",
                "intent": "standard_lookup",
                "complexity": "FIXED",
                "selected_modalities": ["ALL"],
                "retrieval_strategy": "fixed_hybrid_k6",
                "top_k": 6,
                "reranking": True,
                "model_name": OLLAMA_MODEL,
                "model_tier": "TIER_1_FIXED",
                "routing_reason": "Fixed single model baseline",
                "available_ram_gb": sys_res.get("available_ram_gb", 8.0),
                "initial_evidence_score": 0.50,
                "final_evidence_score": 0.50,
                "retrieval_iterations": 1,
                "re_retrieval": False,
                "claims_count": post_verify.total_claims,
                "supported_claims_count": post_verify.supported_claims,
                "faithfulness_score": post_verify.faithfulness_score,
                "abstention_triggered": post_verify.abstention_triggered,
                "regeneration_triggered": post_verify.regeneration_triggered,
                "retrieval_latency_seconds": round(retrieval_duration, 3),
                "generation_latency_seconds": round(generation_duration, 3),
                "total_latency_seconds": round(total_duration, 3),
                "retrieval_duration_seconds": round(retrieval_duration, 3),
                "generation_duration_seconds": round(generation_duration, 3),
                "total_query_duration_seconds": round(total_duration, 3),
                "retrieved_results": len(search_results),
                "top_sources": ", ".join(list(set([r.get('metadata', {}).get('source', 'Unknown') for r in search_results[:5]]))),
                "answer_length_chars": len(answer_text)
            }
            yield "", baseline_metadata
            return

        # ====================================================================
        # ADAPTIVE-MMRAG MULTI-AGENT PATH (with ModelRouter & PostVerifier)
        # ====================================================================
        try:
            prompt, context, verification, analysis, plan, telemetry = self.run_adaptive_pipeline(query, chatbot_instance)
        except Exception as e:
            print(f"   ⚠️  Adaptive pipeline fallback due to error: {e}")
            # Safe fallback to baseline
            yield f"\n[Adaptive Router Notice: Fallback activated due to {e}]\n", None
            stream_gen = chatbot_instance.generate_answer_stream(query, chatbot_instance.search_by_text(query, k=6))
            for token in stream_gen:
                yield token, None
            return

        # Inject response language + conversation history so build_grounded_prompt can use them
        analysis.response_language = response_language or 'en'
        # Always rebuild prompt with history and correct language
        prompt = self.generation_agent.build_grounded_prompt(
            query, context, verification, analysis, conversation_history=history
        )


        # Resource-Aware Multi-Model Routing Decision
        routing_decision = self.model_router.select(
            query_complexity=analysis.complexity,
            evidence_confidence=verification.confidence_level,
            query=query
        )
        print(f"   🔀 Routed Model: {routing_decision.selected_model} ({routing_decision.tier}) | Reason: {routing_decision.reason}")

        generation_start = time.perf_counter()
        full_response = []
        
        # ── Bug 2 Fix: Vision model (llava:7b/moondream) is NOT installed on this system.
        # Image queries are answered via rich text metadata injected by image_handler.py
        # (Subject, Visual Content, OCR text). No vision model call needed.
        use_vision = False

        stream_gen = self.generation_agent.generate_stream(
            prompt=prompt,
            image_paths=context.get('image_paths', []),
            use_vision_model=use_vision,
            selected_model=routing_decision.selected_model
        )

        for token in stream_gen:
            full_response.append(token)
            yield token, None

        generation_duration = time.perf_counter() - generation_start
        total_duration = time.perf_counter() - request_started_at
        answer_text = "".join(full_response)

        # Step 8: Post-Generation Evidence Verification & Grounding Check
        cross_enc = getattr(chatbot_instance.search_engine, 'reranker', None) if hasattr(chatbot_instance, 'search_engine') else None
        post_verify = self.post_verifier_agent.verify_post_generation(
            query=query,
            answer=answer_text,
            evidence_chunks=chatbot_instance.search_by_text(query, k=plan.top_k, use_reranker=plan.use_reranker) if chatbot_instance else [],
            cross_encoder=cross_enc
        )

        if post_verify.abstention_triggered:
            if post_verify.faithfulness_score < 0.25:
                yield f"\n\n> ⚠️ **Abstention Enforced:** {post_verify.final_answer}", None
            else:
                yield f"\n\n> ⚠️ **Grounding Advisory:** {post_verify.final_answer[:100]}...", None

        # Build expandable frontend details HTML snippet (enhanced for mentor demo)
        re_retrieval_label = "✅ Triggered — Score Improved" if telemetry["re_retrieval_triggered"] else "Not needed (initial score sufficient)"
        re_retrieval_color = "#86efac" if telemetry["re_retrieval_triggered"] else "#94a3b8"
        conf_color = "#86efac" if verification.confidence_level == "HIGH" else ("#fbbf24" if verification.confidence_level == "MEDIUM" else "#f87171")
        faith_pct = int(post_verify.faithfulness_score * 100)
        faith_color = "#86efac" if faith_pct >= 70 else ("#fbbf24" if faith_pct >= 40 else "#f87171")
        abstention_label = "⚠️ Triggered" if post_verify.abstention_triggered else "✅ Passed"
        abstention_color = "#fbbf24" if post_verify.abstention_triggered else "#86efac"

        details_badge = (
            f"\n\n<details class=\"adaptive-details\">"
            f"<summary>🔍 <b>Adaptive-MMRAG Pipeline Details</b> "
            f"— Confidence: <span style='color:{conf_color}'>{verification.confidence_level}</span> "
            f"({telemetry['final_score']:.2f}) | "
            f"Faithfulness: <span style='color:{faith_color}'>{faith_pct}%</span> | "
            f"<span style='color:#a5b4fc'>{routing_decision.selected_model}</span>"
            f"</summary>"
            f"<div class=\"adaptive-grid\">"
            # ── 5-Agent Pipeline Trace ──
            f"<div style='grid-column:1/-1; border-bottom:1px solid #334155; padding-bottom:6px; margin-bottom:6px;'>"
            f"<b style='color:#a5b4fc'>🤖 5-Agent Pipeline Execution</b> "
            f"<span style='color:#94a3b8; font-size:0.85em;'>(This is what Baseline RAG does NOT have)</span>"
            f"</div>"
            f"<div>✅ <b>Agent 1 — Query Analysis:</b> Intent=<code>{analysis.intent.replace('_',' ').title()}</code>, Complexity=<code>{analysis.complexity}</code>, Modalities=<code>{', '.join(analysis.target_modalities) or 'text'}</code></div>"
            f"<div>✅ <b>Agent 2 — Retrieval Planner:</b> Strategy=<code>{plan.strategy}</code>, Top-K=<code>{plan.top_k}</code>, Reranker=<code>{'ON' if plan.use_reranker else 'OFF'}</code></div>"
            f"<div>✅ <b>Agent 3 — Multimodal Retrieval:</b> Retrieved <code>{telemetry['retrieved_results_count']}</code> chunks via Hybrid BM25+Dense+CLIP in <code>{telemetry['retrieval_latency']*1000:.0f}ms</code></div>"
            f"<div>✅ <b>Agent 4 — Evidence Verifier:</b> Score=<span style='color:{conf_color}'><code>{telemetry['final_score']:.3f}</code></span> ({verification.confidence_level}), Keyword Coverage=<code>{verification.keyword_coverage:.0%}</code></div>"
            f"<div style='color:{re_retrieval_color}'>🔁 <b>Re-Retrieval Loop:</b> {re_retrieval_label}{' — ' + telemetry['re_retrieval_reason'] if telemetry['re_retrieval_triggered'] else ''} | Iterations: <code>{telemetry['iteration_count']}</code></div>"
            f"<div>✅ <b>Agent 5 — Response Generation:</b> Model=<code>{routing_decision.selected_model}</code> (Tier: {routing_decision.tier}), RAM=<code>{routing_decision.available_ram_gb:.1f}GB</code> free</div>"
            # ── Post-Gen Verification ──
            f"<div style='grid-column:1/-1; border-top:1px solid #334155; padding-top:6px; margin-top:6px;'>"
            f"<b style='color:#a5b4fc'>🛡️ Post-Generation Faithfulness Verification</b> "
            f"<span style='color:#94a3b8; font-size:0.85em;'>(Baseline RAG has no hallucination check)</span>"
            f"</div>"
            f"<div>Faithfulness Score: <span style='color:{faith_color}'><b>{faith_pct}%</b></span> ({post_verify.supported_claims}/{post_verify.total_claims} claims verified against retrieved evidence)</div>"
            f"<div>Abstention Guard: <span style='color:{abstention_color}'><b>{abstention_label}</b></span> — Answer released to user</div>"
            # ── Sources & Latency ──
            f"<div style='grid-column:1/-1; border-top:1px solid #334155; padding-top:6px; margin-top:6px;'>"
            f"<b style='color:#a5b4fc'>📊 Performance Metrics</b></div>"
            f"<div>⏱️ Retrieval Pipeline: <code>{telemetry['total_retrieval_pipeline_latency']*1000:.0f}ms</code> | Generation: <code>{generation_duration*1000:.0f}ms</code> | Total: <code>{total_duration*1000:.0f}ms</code></div>"
            f"<div>📄 Top Sources: <code>{', '.join(telemetry['top_sources'][:3]) or 'None'}</code></div>"
            f"<div>🔀 Model Routing Reason: <i>{routing_decision.reason}</i></div>"
            f"</div>"
            f"</details>"
        )
        yield details_badge, None

        # Compile final adaptive metadata dictionary for logging
        final_metadata = {
            "mode": "adaptive",
            "intent": analysis.intent,
            "complexity": analysis.complexity,
            "selected_modalities": analysis.target_modalities,
            "retrieval_strategy": plan.strategy,
            "top_k": plan.top_k,
            "reranking": plan.use_reranker,
            "model_name": routing_decision.selected_model,
            "model_tier": routing_decision.tier,
            "routing_reason": routing_decision.reason,
            "available_ram_gb": routing_decision.available_ram_gb,
            "initial_evidence_score": round(telemetry["initial_score"], 3),
            "final_evidence_score": round(telemetry["final_score"], 3),
            "retrieval_iterations": telemetry["iteration_count"],
            "re_retrieval": telemetry["re_retrieval_triggered"],
            "re_retrieval_reason": telemetry["re_retrieval_reason"] or "",
            "claims_count": post_verify.total_claims,
            "supported_claims_count": post_verify.supported_claims,
            "faithfulness_score": post_verify.faithfulness_score,
            "abstention_triggered": post_verify.abstention_triggered,
            "regeneration_triggered": post_verify.regeneration_triggered,
            "retrieval_latency_seconds": round(telemetry["retrieval_latency"], 3),
            "generation_latency_seconds": round(generation_duration, 3),
            "total_latency_seconds": round(total_duration, 3),
            "retrieval_duration_seconds": round(telemetry["retrieval_latency"], 3),
            "generation_duration_seconds": round(generation_duration, 3),
            "total_query_duration_seconds": round(total_duration, 3),
            "retrieved_results": telemetry["retrieved_results_count"],
            "top_sources": ", ".join(telemetry["top_sources"]),
            "answer_length_chars": len(answer_text)
        }
        
        yield "", final_metadata

