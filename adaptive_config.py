"""
adaptive_config.py - Centralized configuration for Adaptive-MMRAG framework.
Maintains all thresholds, policies, model settings, and execution toggles.
"""

import os
from typing import Dict, List, Any

# ============================================================================
# SYSTEM MODES & EXECUTION CONTROLS
# ============================================================================
# Set to True for Adaptive-MMRAG (default), or False for Baseline fixed RAG
ADAPTIVE_MODE_DEFAULT: bool = os.getenv("ADAPTIVE_MODE", "true").lower() in ("true", "1", "yes")

# Maximum re-retrieval feedback loops to prevent infinite cycles
MAX_RETRIEVAL_ITERATIONS: int = int(os.getenv("MAX_RETRIEVAL_ITERATIONS", "2"))

# Reranker global toggle
RERANKER_ENABLED_DEFAULT: bool = True

# Resource / latency awareness toggle
RESOURCE_AWARENESS_ENABLED: bool = True


# ============================================================================
# EVIDENCE VERIFICATION THRESHOLDS
# ============================================================================
# Normalized 0.0 - 1.0 confidence score thresholds
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.75,
    "MEDIUM": 0.50,
    "LOW": 0.00
}

# Score below which re-retrieval is triggered (if iterations allow)
RE_RETRIEVAL_THRESHOLD: float = 0.50


# ============================================================================
# ADAPTIVE TOP-K POLICIES
# ============================================================================
TOP_K_POLICIES = {
    "LOW": 4,          # Direct factual lookups (fastest prefill)
    "MEDIUM": 5,       # Standard questions
    "HIGH": 8,         # Complex multi-hop reasoning, comparisons
    "OVERVIEW": 12,    # Document summary and comprehensive overviews
    "RE_RETRIEVE_EXPANSION": 1.4  # Multiplier applied during re-retrieval
}


# ============================================================================
# INTENT & COMPLEXITY DEFINITIONS
# ============================================================================
VALID_INTENTS = [
    "factual",
    "summarization",
    "comparison",
    "visual_question",
    "audio_question",
    "cross_modal",
    "reasoning",
    "document_overview",
    "keyword_lookup"
]

VALID_COMPLEXITIES = ["LOW", "MEDIUM", "HIGH"]

VALID_MODALITIES = ["TEXT", "IMAGE", "AUDIO", "TABLE", "MULTIMODAL"]


# ============================================================================
# LOCAL MODEL CONFIGURATIONS & MULTI-MODEL TIERS
# ============================================================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
# Vision model — only set if you have moondream/llava installed via: ollama pull moondream
# Defaults to the same text model so image queries use text-grounded answers (no 404 errors)
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5:3b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Multi-model routing tiers
# Tier 1: Lightweight / fast model for low complexity queries or constrained RAM
OLLAMA_MODEL_TIER1 = os.getenv("OLLAMA_MODEL_TIER1", "qwen2.5:3b")
# Tier 2: Higher capacity / reasoning model for high complexity, multi-document queries
OLLAMA_MODEL_TIER2 = os.getenv("OLLAMA_MODEL_TIER2", "llama3.1:8b")

# System resource thresholds for model routing
RAM_THRESHOLD_TIER2_GB: float = float(os.getenv("RAM_THRESHOLD_TIER2_GB", "6.0"))

# ============================================================================
# POST-GENERATION EVIDENCE VERIFICATION & ABSTENTION CONFIGURATION
# ============================================================================
# Faithfulness threshold below which abstention / notification is triggered
# Lowered from 0.60 → 0.35: reduces over-aggressive abstention on general/knowledge queries
VERIFICATION_FAITHFULNESS_THRESHOLD: float = float(os.getenv("VERIFICATION_FAITHFULNESS_THRESHOLD", "0.35"))
ABSTENTION_MESSAGE = "No supporting information was found in the available documents for one or more parts of this answer."
FULL_ABSTENTION_MESSAGE = "No supporting information was found in the available documents."

OLLAMA_OPTIONS: Dict[str, Any] = {
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "768")),
    "num_thread": int(os.getenv("OLLAMA_NUM_THREAD", "6")),  # Optimal 6 P-cores for i7-13620H eliminates thread contention
    "temperature": 0.1,
    "repeat_penalty": 1.15,
    "top_p": 0.9,
}

