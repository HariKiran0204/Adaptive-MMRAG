

#!/usr/bin/env python3
"""
rag_chatbot.py - Complete Cross-Modal Multimodal RAG Chatbot
Supports text, image, audio queries with LLaVA vision integration
"""

import os
import numpy as np
import ollama
from typing import List, Dict, Tuple, Optional
from PIL import Image
from pathlib import Path
from vectorstore import (
    UnifiedEmbeddingGenerator,
    get_unified_embedding_generator, 
    load_faiss_index, 
    BM25HybridSearchEngine
)

# ============================================================================
# CONFIGURATION
# ============================================================================oll
FAISS_INDEX_PATH = "processed_output/multimodal_index.bin"
# Default text model is qwen2.5:3b and default vision model is moondream (or llava:7b).
# Both models can be overridden via environment variables without editing code.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")
OLLAMA_OPTIONS = {
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "768")),
    "num_thread": int(os.getenv("OLLAMA_NUM_THREAD", "6")),  # Optimal 6 P-cores on i7-13620H
    "temperature": 0.1,
    "repeat_penalty": 1.15,
    "top_p": 0.9,
}
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")



# ============================================================================
# CROSS-MODAL RAG CHATBOT
# ============================================================================

class CrossModalRAGChatbot:
    """
    Complete Cross-Modal RAG Chatbot with vision support.
    
    Features:
    - Text queries → All modalities
    - Image queries → Similar images + related content
    - Audio queries → Related transcripts + content
    - Multi-modal queries → Fused results
    - LLaVA vision integration for image understanding
    - Citation transparency with source tracking
    """
    
    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        """Initialize the chatbot with vector index."""
        self.index_path = index_path
        
        self._print_header()
        self._load_components()
        self._print_ready()
    
    def _print_header(self):
        """Print startup header."""
        print(f"\n{'='*70}")
        print(f"🤖 CROSS-MODAL MULTIMODAL RAG CHATBOT")
        print(f"{'='*70}")
        print(f"Model: {OLLAMA_MODEL}")
        print(f"Index: {self.index_path}")
        print(f"\n🎯 Supported Query Modes:")
        print(f"   ✅ Text → Text, Images, Audio, Tables")
        print(f"   ✅ Image → Similar Images + Related Content")
        print(f"   ✅ Audio → Related Transcripts + Content")
        print(f"   ✅ Multi-modal → Fused Results")
    
    def _load_components(self):
        """Load vector index and search engine."""
        print(f"\n🔥 Loading components...")
        
        # Load FAISS index and metadata
        self.index, self.metadata = load_faiss_index(self.index_path)
        
        # Initialize embedding generator
        self.generator = get_unified_embedding_generator()
        
        # Initialize hybrid search engine
        print("🔍 Initializing hybrid search engine...")
        self.search_engine = BM25HybridSearchEngine(
            self.index, self.metadata, self.generator
        )
    
    def _print_ready(self):
        """Print ready status."""
        print(f"✅ Ready! {len(self.metadata)} chunks indexed")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # SEARCH METHODS
    # ========================================================================
    
    def is_summary_query(self, query: str) -> bool:
        """Identify if a user is asking a general summary or overview question."""
        q = query.lower()
        summary_keywords = [
            "summarize", "summary", "overview", "what is present", "what is in", 
            "what does the document contain", "topics covered", "about this document",
            "tell me about this file", "explain the document", "explain the file",
            "describe the document", "general idea"
        ]
        return any(keyword in q for keyword in summary_keywords)

    def search_by_text(
        self, 
        query: str, 
        k: int = 10, 
        use_reranker: bool = True
    ) -> List[Dict]:
        """
        Text query → Multi-modal results.
        
        Args:
            query: Text query string
            k: Number of results to return
            use_reranker: Whether to use cross-encoder reranking
            
        Returns:
            List of search results with scores and metadata
        """
        # If it is a general summary query, return a uniform sample of chunks across the entire document
        if self.is_summary_query(query) and self.metadata:
            print("   ℹ️  Summary query detected. Generating representative document overview context...")
            num_samples = min(20, len(self.metadata))
            indices = [int(i * (len(self.metadata) / num_samples)) for i in range(num_samples)]
            results = []
            for i, idx in enumerate(indices):
                results.append({
                    'score': 1.0,
                    'rank': i + 1,
                    'metadata': self.metadata[idx]
                })
            return results

        return self.search_engine.hybrid_search(
            query, k=k, use_reranker=use_reranker
        )
    
    def search_by_image(self, image_path: str, k: int = 10) -> List[Dict]:
        """
        Image query → Similar images + related content.
        
        Uses CLIP image embeddings to find:
        - Visually similar images
        - Related text mentioning similar concepts
        - Related tables and audio
        
        Args:
            image_path: Path to query image
            k: Number of results to return
            
        Returns:
            List of search results
        """
        print(f"🖼️  Searching by image: {os.path.basename(image_path)}")
        
        try:
            # Load and embed query image
            image = Image.open(image_path).convert('RGB')
            query_embedding = self.generator.embed(image)
            
            # Ensure correct format for FAISS
            if query_embedding.ndim == 1:
                query_embedding = np.array([query_embedding])
            query_embedding = query_embedding.astype(np.float32)
            
            # Search FAISS index
            similarities, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx != -1:
                    results.append({
                        'score': float(similarities[0][i]),
                        'rank': i + 1,
                        'metadata': self.metadata[idx]
                    })
            
            print(f"✅ Found {len(results)} similar items")
            return results
        
        except Exception as e:
            print(f"❌ Image search failed: {e}")
            return []
    
    def search_by_audio(self, audio_path: str, k: int = 10) -> List[Dict]:
        """
        Audio query → Related transcripts + content.
        
        Process:
        1. Transcribe audio file
        2. Use transcript as text query
        3. Find related content
        
        Args:
            audio_path: Path to audio file
            k: Number of results to return
            
        Returns:
            List of search results
        """
        print(f"🎧 Searching by audio: {os.path.basename(audio_path)}")
        
        try:
            from audio_handler import EnhancedAudioHandler, AudioConfig
            
            # Transcribe audio
            print("   🔄 Transcribing audio...")
            config = AudioConfig(num_speakers=None)
            handler = EnhancedAudioHandler(config)
            chunks = handler.process_audio(audio_path)
            
            if not chunks:
                print("   ❌ Transcription failed")
                return []
            
            # Extract transcript
            transcript = ' '.join([c['content'] for c in chunks])
            print(f"   ✅ Transcribed {len(transcript)} characters")
            
            # Search using transcript
            print("   🔍 Searching with transcript...")
            return self.search_by_text(transcript, k=k, use_reranker=False)
        
        except Exception as e:
            print(f"❌ Audio search failed: {e}")
            return []
    
    def search_multimodal(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        k: int = 10
    ) -> List[Dict]:
        """
        Multi-modal query → Fused results.
        
        Combines results from multiple query types using
        Reciprocal Rank Fusion (RRF).
        
        Args:
            text_query: Optional text query
            image_path: Optional image path
            audio_path: Optional audio path
            k: Number of results to return
            
        Returns:
            Fused search results
        """
        print(f"🔄 Multi-modal search:")
        if text_query:
            print(f"   📝 Text: {text_query[:50]}...")
        if image_path:
            print(f"   🖼️  Image: {os.path.basename(image_path)}")
        if audio_path:
            print(f"   🎧 Audio: {os.path.basename(audio_path)}")
        
        all_results = []
        
        # Text search
        if text_query:
            print("\n   1️⃣ Performing text search...")
            text_results = self.search_engine.semantic_search(text_query, k=k)
            all_results.append(text_results)
        
        # Image search
        if image_path:
            print("   2️⃣ Performing image search...")
            image_results = self.search_by_image(image_path, k=k)
            image_results_tuples = [
                (r['rank'] - 1, r['score']) for r in image_results
            ]
            all_results.append(image_results_tuples)
        
        # Audio search
        if audio_path:
            print("   3️⃣ Performing audio search...")
            audio_results = self.search_by_audio(audio_path, k=k)
            audio_results_tuples = [
                (r['rank'] - 1, r['score']) for r in audio_results
            ]
            all_results.append(audio_results_tuples)
        
        if not all_results:
            return []
        
        # Fuse results using RRF
        print("   4️⃣ Fusing results with Reciprocal Rank Fusion...")
        fused = self.search_engine.reciprocal_rank_fusion(all_results, k=60)
        
        # Format fused results
        final_results = []
        for i, (doc_idx, score) in enumerate(fused[:k]):
            final_results.append({
                'score': score,
                'rank': i + 1,
                'metadata': self.metadata[doc_idx]
            })
        
        print(f"   ✅ Fused to {len(final_results)} results\n")
        return final_results
    
    # ========================================================================
    # CONTEXT FORMATTING
    # ========================================================================
    
    def format_context_by_modality(self, search_results: List[Dict]) -> Dict:
        """
        Separate and format context by modality.
        
        Args:
            search_results: List of search results
            
        Returns:
            Dictionary with context organized by modality
        """
        context = {
            'text': [],
            'tables': [],
            'images': [],
            'image_paths': [],
            'audio': []
        }
        
        for result in search_results:
            meta = result['metadata']
            orig = meta.get('original_metadata', {})
            
            modality = meta.get('modality') or orig.get('modality') or meta.get('type') or orig.get('type', 'text')
            content_type = meta.get('type') or orig.get('type', 'text')
            content_str = meta.get('content') or orig.get('content', '')
            source_name = meta.get('source') or orig.get('source', 'Unknown')
            
            # Audio content
            if modality == 'audio' or content_type == 'audio':
                context['audio'].append({
                    'content': content_str,
                    'speaker': meta.get('speaker') or orig.get('speaker', 'Unknown'),
                    'timestamp': meta.get('timestamp_formatted') or orig.get('timestamp_formatted', 'N/A'),
                    'score': result['score'],
                    'source': source_name
                })
            
            # Image content
            elif modality == 'image' or content_type == 'image':
                context['images'].append({
                    'content': content_str,
                    'score': result['score'],
                    'source': source_name
                })
                
                # Check both meta and orig for the actual image file path
                img_path = meta.get('image_path') or orig.get('image_path')
                if img_path and os.path.exists(img_path) and img_path not in context['image_paths']:
                    context['image_paths'].append(img_path)
            
            # Table content
            elif modality == 'table' or content_type == 'table':
                context['tables'].append({
                    'content': content_str,
                    'title': meta.get('table_title') or orig.get('table_title', 'Untitled Table'),
                    'score': result['score'],
                    'source': source_name
                })
            
            # Text content
            else:
                context['text'].append({
                    'content': content_str,
                    'source': source_name,
                    'score': result['score']
                })
        
        return context
    
    def build_comprehensive_prompt(
        self, 
        query: str, 
        context: Dict, 
        query_type: str = "text"
    ) -> str:
        """
        Build a comprehensive prompt with all modalities.
        
        Args:
            query: User query
            context: Context dictionary by modality
            query_type: Type of query (text/image/audio/multimodal)
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # System message
        parts.append("You are an expert AI assistant analyzing multimodal content.")
        
        # Query type context
        if query_type == "image":
            parts.append("The user uploaded an image to find similar content.")
        elif query_type == "audio":
            parts.append("The user uploaded an audio file to find related content.")
        elif query_type == "multimodal":
            parts.append("The user provided multiple query types (text/image/audio).")
        
        parts.append("\n" + "="*70)
        
        # Text context
        if context['text']:
            parts.append("\n=== TEXT CONTEXT ===")
            for i, item in enumerate(context['text'][:5], 1):
                parts.append(f"\n[Text {i}] (Relevance: {item['score']:.3f})")
                parts.append(f"Source: {item['source']}")
                parts.append(item['content'][:500])
        
        # Table context
        if context['tables']:
            parts.append("\n\n=== TABLE DATA ===")
            for i, item in enumerate(context['tables'][:3], 1):
                parts.append(f"\n[Table {i}] {item['title']} (Relevance: {item['score']:.3f})")
                parts.append(f"Source: {item['source']}")
                parts.append(item['content'][:800])
        
        # Image context
        if context['images']:
            parts.append(f"\n\n=== IMAGES ===")
            parts.append(f"{len(context['image_paths'])} relevant images are being analyzed.")
            for i, item in enumerate(context['images'][:3], 1):
                parts.append(f"\n[Image {i}] (Relevance: {item['score']:.3f})")
                parts.append(f"Source: {item['source']}")
                parts.append(item['content'][:300])
        
        # Audio context
        if context['audio']:
            parts.append("\n\n=== AUDIO TRANSCRIPTS ===")
            for i, item in enumerate(context['audio'][:3], 1):
                parts.append(
                    f"\n[Audio {i}] {item['speaker']} at {item['timestamp']} "
                    f"(Relevance: {item['score']:.3f})"
                )
                parts.append(f"Source: {item['source']}")
                parts.append(item['content'][:400])
        
        # Question
        parts.append("\n\n" + "="*70)
        parts.append(f"\n=== QUESTION ===")
        parts.append(query)
        
        # Instructions
        parts.append("\n\n=== CORE PRINCIPLES ===")
        core_principles = [
            "1. **Answer ONLY from the provided context** - Never use external knowledge",
            "2. **Be thorough yet concise** - Include all relevant details without redundancy",
            "3. **Cite specific evidence** - Reference exact figures, tables, quotes, or data points"
        ]
        parts.extend(core_principles)
        
        parts.append("\n\n=== RESPONSE GUIDELINES ===")
        parts.append("\n**For Summaries:**")
        summary_guidelines = [
            "- Start with a brief 1-2 sentence overview",
            "- Organize information into clear sections with headers",
            "- Highlight key findings, main arguments, or critical data",
            "- Include specific numbers, percentages, or metrics when present",
            "- End with main takeaways or implications"
        ]
        parts.extend(summary_guidelines)
        
        parts.append("\n\n**For Specific Questions:**")
        question_guidelines = [
            "- Answer the exact question asked - no more, no less",
            "- Lead with the direct answer, then provide supporting details",
            "- Use bullet points for lists or multiple items",
            "- Reference figure/table numbers when citing visual data"
        ]
        parts.extend(question_guidelines)
        
        parts.append("\n\n**For Image Analysis:**")
        image_guidelines = [
            "- Describe what the image shows in detail",
            "- Identify key components, labels, and relationships",
            "- Explain the significance or purpose of the visual",
            "- Connect image content to surrounding text context"
        ]
        parts.extend(image_guidelines)
        
        parts.append("\n\n**For Table Analysis:**")
        table_guidelines = [
            "- Identify what the table measures or compares",
            "- Highlight notable patterns, trends, or outliers",
            "- Present key statistics or comparisons clearly",
            "- Explain what the data reveals or supports"
        ]
        parts.extend(table_guidelines)
        
        parts.append("\n\n=== FORMATTING RULES ===")
        formatting_rules = [
            "- Use **bold** for emphasis on key terms or findings",
            "- Use headers (##) to organize longer responses",
            "- Use bullet points (-) for lists of items",
            "- Use numbered lists (1.) for sequential steps or rankings",
            "- Keep paragraphs focused and digestible (3-5 sentences max)"
        ]
        parts.extend(formatting_rules)
        
        parts.append("\n\n=== WHAT TO AVOID ===")
        avoid_rules = [
            "- Never mention \"chunk\", \"score\", \"metadata\", \"relevance score\",or \"context\" in your answer",
            "- Don't use phrases like \"according to the provided context\"",
            "- Avoid vague statements - be specific with evidence",
            "- Don't add information not present in the context",
            "- Don't repeat the question back unnecessarily"
        ]
        parts.extend(avoid_rules)
        
        parts.append("\n=== DETAILED ANSWER ===")
        
        return "\n".join(parts)
    
    # ========================================================================
    # ANSWER GENERATION
    # ========================================================================

    def build_fast_rag_prompt(self, query: str, context: Dict) -> str:
        """Build a compact evidence prompt for responsive local inference."""
        evidence = []
        
        is_summary = self.is_summary_query(query)
        max_text_chunks = 15 if is_summary else 4
        max_char_len = 4000 if is_summary else 3000
        max_evidence_len = 12000 if is_summary else 8000

        for item in context['text'][:max_text_chunks]:
            evidence.append(f"Source: {item['source']}\n{item['content'][:max_char_len]}")
        for item in context['tables'][:2]:
            evidence.append(
                f"Source: {item['source']} (table: {item['title']})\n"
                f"{item['content'][:3000]}"
            )
        for item in context['images'][:2]:
            evidence.append(f"Source: {item['source']} (image description)\n{item['content'][:2000]}")
        for item in context['audio'][:2]:
            evidence.append(f"Source: {item['source']} (audio transcript)\n{item['content'][:2000]}")

        # Keep prompt evaluation bounded on CPU while retaining the strongest
        # retrieved evidence.
        evidence_text = "\n\n".join(evidence)[:max_evidence_len]
        
        if is_summary:
            instruction = (
                "Provide a comprehensive, high-level summary of the document contents based on the evidence below. "
                "List the main topics, sections, and types of questions (like algorithms, code, space/time complexities) "
                "covered in the document. Respond in bullet points. Cite the source filename. Do not mention chunks or scores.\n\n"
            )
        elif context.get('images') or context.get('image_paths'):
            instruction = (
                "You are an expert multimodal assistant. Examine the document and image evidence below carefully.\n"
                "1. If the image is a photo or visual object (e.g. apple, object, vehicle), state clearly and directly what it depicts in the first sentence.\n"
                "2. If the image is a certificate, document, or form, extract all exact names, dates, registration/ID numbers, and credentials.\n"
                "3. Provide a direct, factual, and crisp answer based strictly on the evidence.\n"
                "4. Cite the source filename at the end. Do not mention chunks or retrieval scores.\n\n"
            )
        else:
            instruction = (
                "Answer the question using only the document evidence below. Give a direct, "
                "clear explanation with the important details. Cite the source filename. If the "
                "evidence is insufficient, say so. Do not mention chunks or retrieval scores.\n\n"
            )

        return f"{instruction}DOCUMENT EVIDENCE:\n{evidence_text}\n\nQUESTION: {query}\n\nANSWER:"
    
    def generate_answer(
        self, 
        query: str, 
        search_results: List[Dict], 
        query_type: str = "text"
    ) -> str:
        """
        Generate answer using multimodal context with LLaVA.
        
        Args:
            query: User query
            search_results: List of search results
            query_type: Type of query
            
        Returns:
            Generated answer string
        """
        # Format context
        context = self.format_context_by_modality(search_results)
        
        # Build prompt
        prompt = self.build_fast_rag_prompt(query, context)
        
        try:
            # Only route to heavy vision model if visual appearance/colors/photo is specifically requested,
            # or if query_type is explicitly 'image', or if no OCR text could be extracted.
            pure_visual_terms = (
                "color", "colour", "appearance", "look like", "photo of",
                "picture of", "what does the person look like", "visual appearance"
            )
            has_extracted_text = any(len(img.get('content', '')) > 20 for img in context['images']) or len(context['text']) > 0
            
            vision_requested = (
                query_type in {"image", "multimodal"}
                or (any(term in query.lower() for term in pure_visual_terms) and not has_extracted_text)
                or (len(context['image_paths']) > 0 and not has_extracted_text)
            )

            if context['image_paths'] and vision_requested:
                print(f"   🖼️  Analyzing {len(context['image_paths'])} images with Vision Model ({OLLAMA_VISION_MODEL})...")
                try:
                    response = ollama.chat(
                        model=OLLAMA_VISION_MODEL,
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': context['image_paths'][:5]  # Limit to 5 images
                        }],
                        options=OLLAMA_OPTIONS,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                    return response['message']['content'].strip()
                except Exception as vision_err:
                    if "not found" in str(vision_err).lower() and OLLAMA_VISION_MODEL != "llava:7b":
                        print(f"   ⚠️  Vision model '{OLLAMA_VISION_MODEL}' not found. Falling back to 'llava:7b'...")
                        response = ollama.chat(
                            model="llava:7b",
                            messages=[{
                                'role': 'user',
                                'content': prompt,
                                'images': context['image_paths'][:5]
                            }],
                            options=OLLAMA_OPTIONS,
                            keep_alive=OLLAMA_KEEP_ALIVE,
                        )
                        return response['message']['content'].strip()
                    raise vision_err
            else:
                # Text-only generation
                print(f"   📝 Generating answer...")
                response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    options=OLLAMA_OPTIONS,
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                return response.get('response', 'No response generated.').strip()
        
        except Exception as e:
            return f"❌ Error: {e}\nEnsure Ollama is running: ollama serve"
    
    def generate_answer_stream(
        self, 
        query: str, 
        search_results: List[Dict], 
        query_type: str = "text"
    ):
        """
        Generate answer using multimodal context in streaming mode.
        
        Args:
            query: User query
            search_results: List of search results
            query_type: Type of query
            
        Yields:
            Response tokens
        """
        # Format context
        context = self.format_context_by_modality(search_results)
        
        # Build prompt
        prompt = self.build_fast_rag_prompt(query, context)
        
        try:
            # Only route to heavy vision model if visual appearance/colors/photo is specifically requested,
            # or if query_type is explicitly 'image', or if no OCR text could be extracted.
            pure_visual_terms = (
                "color", "colour", "appearance", "look like", "photo of",
                "picture of", "what does the person look like", "visual appearance"
            )
            has_extracted_text = any(len(img.get('content', '')) > 20 for img in context['images']) or len(context['text']) > 0
            
            vision_requested = (
                query_type in {"image", "multimodal"}
                or (any(term in query.lower() for term in pure_visual_terms) and not has_extracted_text)
                or (len(context['image_paths']) > 0 and not has_extracted_text)
            )

            if context['image_paths'] and vision_requested:
                print(f"   🖼️  Analyzing {len(context['image_paths'])} images with Vision Model ({OLLAMA_VISION_MODEL}) (stream)...")
                try:
                    response_stream = ollama.chat(
                        model=OLLAMA_VISION_MODEL,
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': context['image_paths'][:5]
                        }],
                        stream=True,
                        options=OLLAMA_OPTIONS,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                    for chunk in response_stream:
                        yield chunk['message']['content']
                except Exception as vision_err:
                    if "not found" in str(vision_err).lower() and OLLAMA_VISION_MODEL != "llava:7b":
                        print(f"   ⚠️  Vision model '{OLLAMA_VISION_MODEL}' not found. Falling back to 'llava:7b'...")
                        response_stream = ollama.chat(
                            model="llava:7b",
                            messages=[{
                                'role': 'user',
                                'content': prompt,
                                'images': context['image_paths'][:5]
                            }],
                            stream=True,
                            options=OLLAMA_OPTIONS,
                            keep_alive=OLLAMA_KEEP_ALIVE,
                        )
                        for chunk in response_stream:
                            yield chunk['message']['content']
                    else:
                        raise vision_err
            else:
                # Text-only generation
                print(f"   📝 Generating answer (stream)...")
                response_stream = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                for chunk in response_stream:
                    yield chunk.get('response', '')
        
        except Exception as e:
            yield f"❌ Error: {e}\nEnsure Ollama is running: ollama serve"

    # ========================================================================
    # CHAT METHODS
    # ========================================================================
    
    def chat_text(self, query: str, k: int = 10) -> str:
        """
        Text query mode.
        
        Args:
            query: Text query
            k: Number of results
            
        Returns:
            Generated answer
        """
        print(f"\n{'='*70}")
        print(f"📝 TEXT QUERY: {query}")
        print(f"{'='*70}")
        
        # Search
        print(f"\n🔍 Searching (Hybrid: Semantic + BM25 + Reranking)...")
        results = self.search_by_text(query, k=k)
        
        if not results:
            return "❌ No relevant information found."
        
        # Show what was retrieved
        context = self.format_context_by_modality(results)
        print(f"\n📊 Retrieved:")
        print(f"   Text chunks: {len(context['text'])}")
        print(f"   Tables: {len(context['tables'])}")
        print(f"   Images: {len(context['images'])} ({len(context['image_paths'])} with files)")
        print(f"   Audio: {len(context['audio'])}")
        
        # Generate answer
        answer = self.generate_answer(query, results, query_type="text")
        
        return answer
    
    def chat_image(
        self, 
        image_path: str, 
        question: str = "What is related to this image?", 
        k: int = 10
    ) -> str:
        """
        Image query mode.
        
        Args:
            image_path: Path to query image
            question: Question about the image
            k: Number of results
            
        Returns:
            Generated answer
        """
        print(f"\n{'='*70}")
        print(f"🖼️  IMAGE QUERY: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        # Search by image
        results = self.search_by_image(image_path, k=k)
        
        if not results:
            return "❌ No similar content found."
        
        # Show what was retrieved
        context = self.format_context_by_modality(results)
        print(f"\n📊 Retrieved:")
        print(f"   Similar images: {len([c for c in context['images']])}")
        print(f"   Related text: {len(context['text'])}")
        print(f"   Related tables: {len(context['tables'])}")
        print(f"   Related audio: {len(context['audio'])}")
        
        # Generate answer
        query = f"Based on the uploaded image, {question}"
        answer = self.generate_answer(query, results, query_type="image")
        
        return answer
    
    def chat_audio(
        self, 
        audio_path: str, 
        question: str = "What is related to this audio?", 
        k: int = 10
    ) -> str:
        """
        Audio query mode.
        
        Args:
            audio_path: Path to audio file
            question: Question about the audio
            k: Number of results
            
        Returns:
            Generated answer
        """
        print(f"\n{'='*70}")
        print(f"🎧 AUDIO QUERY: {os.path.basename(audio_path)}")
        print(f"{'='*70}")
        
        # Search by audio
        results = self.search_by_audio(audio_path, k=k)
        
        if not results:
            return "❌ No related content found."
        
        # Show what was retrieved
        context = self.format_context_by_modality(results)
        print(f"\n📊 Retrieved:")
        print(f"   Related text: {len(context['text'])}")
        print(f"   Related tables: {len(context['tables'])}")
        print(f"   Related images: {len(context['images'])}")
        print(f"   Related audio: {len(context['audio'])}")
        
        # Generate answer
        query = f"Based on the uploaded audio, {question}"
        answer = self.generate_answer(query, results, query_type="audio")
        
        return answer
    
    def chat_multimodal(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        k: int = 10
    ) -> str:
        """
        Multi-modal query mode.
        
        Args:
            text_query: Optional text query
            image_path: Optional image path
            audio_path: Optional audio path
            k: Number of results
            
        Returns:
            Generated answer
        """
        print(f"\n{'='*70}")
        print(f"🔄 MULTI-MODAL QUERY")
        print(f"{'='*70}")
        
        # Search with multiple modalities
        results = self.search_multimodal(
            text_query=text_query,
            image_path=image_path,
            audio_path=audio_path,
            k=k
        )
        
        if not results:
            return "❌ No relevant information found."
        
        # Show what was retrieved
        context = self.format_context_by_modality(results)
        print(f"\n📊 Retrieved:")
        print(f"   Text chunks: {len(context['text'])}")
        print(f"   Tables: {len(context['tables'])}")
        print(f"   Images: {len(context['images'])} ({len(context['image_paths'])} with files)")
        print(f"   Audio: {len(context['audio'])}")
        
        # Generate answer
        query_parts = []
        if text_query:
            query_parts.append(text_query)
        if image_path:
            query_parts.append(f"[Uploaded image: {os.path.basename(image_path)}]")
        if audio_path:
            query_parts.append(f"[Uploaded audio: {os.path.basename(audio_path)}]")
        
        query = " + ".join(query_parts)
        answer = self.generate_answer(query, results, query_type="multimodal")
        
        return answer
    
    # ========================================================================
    # INTERACTIVE MODE
    # ========================================================================
    
    def interactive(self):
        """Enhanced interactive mode with cross-modal support."""
        print(f"\n{'='*70}")
        print(f"💬 INTERACTIVE CROSS-MODAL RAG")
        print(f"{'='*70}")
        print("\nQuery Modes:")
        print("  1. Text: Just type your question")
        print("  2. Image: 'image <path>' or 'img <path>'")
        print("  3. Audio: 'audio <path>' or 'aud <path>'")
        print("  4. Multi-modal: 'multi text:<query> image:<path> audio:<path>'")
        print("\nOther Commands:")
        print("  - 'stats' - Show index statistics")
        print("  - 'help' - Show this help")
        print("  - 'exit' - Quit")
        print(f"{'='*70}\n")
        
        while True:
            try:
                query = input("\n🔍 Your query: ").strip()
                
                if not query:
                    continue
                
                # Exit
                if query.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break
                
                # Help
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                # Stats
                if query.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Image query
                if query.lower().startswith(('image ', 'img ')):
                    self._handle_image_query(query)
                    continue
                
                # Audio query
                if query.lower().startswith(('audio ', 'aud ')):
                    self._handle_audio_query(query)
                    continue
                
                # Multi-modal query
                if query.lower().startswith('multi '):
                    self._handle_multimodal_query(query)
                    continue
                
                # Default: Text query
                answer = self.chat_text(query)
                self._display_answer(answer)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _handle_image_query(self, query: str):
        """Handle image query from interactive mode."""
        image_path = query.split(maxsplit=1)[1].strip()
        
        # Optional question after image
        if ' ' in image_path and not os.path.exists(image_path):
            parts = image_path.rsplit(' ', 1)
            if len(parts) == 2 and os.path.exists(parts[0]):
                image_path = parts[0]
                question = parts[1]
            else:
                question = "What is related to this image?"
        else:
            question = "What is related to this image?"
        
        answer = self.chat_image(image_path, question)
        self._display_answer(answer)
    
    def _handle_audio_query(self, query: str):
        """Handle audio query from interactive mode."""
        audio_path = query.split(maxsplit=1)[1].strip()
        
        # Optional question after audio
        if ' ' in audio_path and not os.path.exists(audio_path):
            parts = audio_path.rsplit(' ', 1)
            if len(parts) == 2 and os.path.exists(parts[0]):
                audio_path = parts[0]
                question = parts[1]
            else:
                question = "What is related to this audio?"
        else:
            question = "What is related to this audio?"
        
        answer = self.chat_audio(audio_path, question)
        self._display_answer(answer)
    
    def _handle_multimodal_query(self, query: str):
        """Handle multi-modal query from interactive mode."""
        params = query[6:].strip()
        
        text_query = None
        image_path = None
        audio_path = None
        
        # Parse parameters
        import re
        text_match = re.search(r'text:([^;]+)', params)
        image_match = re.search(r'image:([^;]+)', params)
        audio_match = re.search(r'audio:([^;]+)', params)
        
        if text_match:
            text_query = text_match.group(1).strip()
        if image_match:
            image_path = image_match.group(1).strip()
        if audio_match:
            audio_path = audio_match.group(1).strip()
        
        if not any([text_query, image_path, audio_path]):
            print("❌ Invalid multi-modal query format")
            print("Example: multi text:your question image:/path/to/img.png")
            return
        
        answer = self.chat_multimodal(
            text_query=text_query,
            image_path=image_path,
            audio_path=audio_path
        )
        self._display_answer(answer)
    
    def _show_help(self):
        """Show help message."""
        print("\n📖 Query Examples:")
        print("  Text: What are the main findings?")
        print("  Image: image /path/to/diagram.png")
        print("  Audio: audio /path/to/recording.mp3")
        print("  Multi: multi text:analyze this image:diagram.png")
    
    def _display_answer(self, answer: str):
        """Display answer with formatting."""
        print(f"\n{'='*70}")
        print(f"🤖 ANSWER:")
        print(f"{'='*70}")
        print(answer)
        print(f"{'='*70}")
    
    def _show_stats(self):
        """Show index statistics."""
        from collections import Counter
        
        types = Counter(m['type'] for m in self.metadata)
        modalities = Counter(m.get('modality', 'text') for m in self.metadata)
        sources = Counter(m['source'] for m in self.metadata)
        
        # Count chunks with images
        chunks_with_images = sum(
            1 for m in self.metadata 
            if m.get('original_metadata', {}).get('has_image_file', False)
        )
        
        print(f"\n{'='*70}")
        print(f"📊 INDEX STATISTICS")
        print(f"{'='*70}")
        print(f"Total Chunks: {len(self.metadata)}")
        print(f"\nBy Type:")
        for typ, count in types.most_common():
            print(f"   {typ}: {count}")
        print(f"\nBy Modality:")
        for mod, count in modalities.most_common():
            print(f"   {mod}: {count}")
        print(f"\nChunks with Images: {chunks_with_images}")
        print(f"Unique Sources: {len(sources)}")
        print(f"\n🎯 Cross-Modal Capabilities:")
        print(f"   ✅ Text queries")
        print(f"   ✅ Image queries")
        print(f"   ✅ Audio queries")
        print(f"   ✅ Multi-modal queries")
        print(f"{'='*70}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function with command-line interface."""
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode
        command = sys.argv[1].lower()
        
        if command == 'text':
            # Text query
            if len(sys.argv) < 3:
                print("Usage: python rag_chatbot.py text <query>")
                return
            
            query = ' '.join(sys.argv[2:])
            chatbot = CrossModalRAGChatbot()
            answer = chatbot.chat_text(query)
            print(f"\n🤖 Answer:\n{answer}\n")
        
        elif command == 'image':
            # Image query
            if len(sys.argv) < 3:
                print("Usage: python rag_chatbot.py image <path> [question]")
                return
            
            image_path = sys.argv[2]
            question = ' '.join(sys.argv[3:]) if len(sys.argv) > 3 else "What is related to this image?"
            chatbot = CrossModalRAGChatbot()
            answer = chatbot.chat_image(image_path, question)
            print(f"\n🤖 Answer:\n{answer}\n")
        
        elif command == 'audio':
            # Audio query
            if len(sys.argv) < 3:
                print("Usage: python rag_chatbot.py audio <path> [question]")
                return
            
            audio_path = sys.argv[2]
            question = ' '.join(sys.argv[3:]) if len(sys.argv) > 3 else "What is related to this audio?"
            chatbot = CrossModalRAGChatbot()
            answer = chatbot.chat_audio(audio_path, question)
            print(f"\n🤖 Answer:\n{answer}\n")
        
        elif command == 'help':
            print_usage()
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for usage information")
    else:
        # Interactive mode
        chatbot = CrossModalRAGChatbot()
        chatbot.interactive()


def print_usage():
    """Print usage instructions."""
    usage = """
╔═══════════════════════════════════════════════════════════════════╗
║          CROSS-MODAL RAG CHATBOT - USAGE GUIDE                    ║
╚═══════════════════════════════════════════════════════════════════╝

🎯 COMMAND-LINE MODES:

1. Text Query:
   python rag_chatbot.py text "Your question here"
   
   Example:
   python rag_chatbot.py text "What are the Q4 revenue numbers?"

2. Image Query:
   python rag_chatbot.py image <path> [optional question]
   
   Example:
   python rag_chatbot.py image diagram.png "What does this show?"

3. Audio Query:
   python rag_chatbot.py audio <path> [optional question]
   
   Example:
   python rag_chatbot.py audio meeting.mp3 "What was discussed?"

4. Interactive Mode (default):
   python rag_chatbot.py
   
   Then use:
   - Text queries: Just type your question
   - Image queries: image /path/to/image.png
   - Audio queries: audio /path/to/audio.mp3
   - Multi-modal: multi text:query image:path.png audio:path.mp3

💡 INTERACTIVE MODE COMMANDS:

Text Query:
  > What are the main findings?

Image Query:
  > image /path/to/screenshot.png
  > img diagram.png what does this show?

Audio Query:
  > audio /path/to/meeting.mp3
  > aud recording.wav what was discussed?

Multi-modal Query:
  > multi text:analyze this image:chart.png
  > multi text:find related content image:screenshot.png audio:meeting.mp3

System Commands:
  > stats    - Show index statistics
  > help     - Show help message
  > exit     - Quit the chatbot

📊 FEATURES:

✅ Cross-Modal Search
   - Text → All modalities
   - Image → Similar images + related content
   - Audio → Related transcripts + content
   - Hybrid search with reranking

✅ Vision Understanding
   - LLaVA model analyzes images directly
   - Not just OCR - true visual understanding
   - Multi-image support (up to 5 images)

✅ Citation Transparency
   - Every answer cites sources
   - File names, timestamps, speakers
   - Relevance scores included

✅ Natural Language
   - Ask questions naturally
   - No special syntax required
   - Context-aware responses

🔧 CONFIGURATION:

Index Path: Set FAISS_INDEX_PATH in script
   Default: processed_output/multimodal_index.bin

LLM Model: Set OLLAMA_MODEL in script
   Default: llava:7b

Before running:
   1. Ensure Ollama is running: ollama serve
   2. Ensure LLaVA is available: ollama pull llava:7b
   3. Process your data first with multimodal_unified_parallel.py

═══════════════════════════════════════════════════════════════════
"""
    print(usage)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR INTEGRATION
# ============================================================================

def generate_answer_with_llava(query: str, search_results: List[Dict]) -> str:
    """
    Convenience function for integration with other scripts.
    
    Args:
        query: User query
        search_results: Search results from hybrid search
        
    Returns:
        Generated answer
    """
    # Create temporary chatbot instance
    chatbot = CrossModalRAGChatbot()
    
    # Generate answer
    return chatbot.generate_answer(query, search_results, query_type="text")


def format_context_with_images(search_results: List[Dict]) -> Tuple[str, List[str]]:
    """
    Convenience function to format context and extract image paths.
    Used by Streamlit app.
    
    Args:
        search_results: Search results
        
    Returns:
        Tuple of (text_context, image_paths)
    """
    chatbot = CrossModalRAGChatbot()
    context = chatbot.format_context_by_modality(search_results)
    
    # Build text context
    context_parts = []
    
    if context['text']:
        context_parts.append("=== TEXT CONTENT ===\n")
        for i, item in enumerate(context['text'], 1):
            context_parts.append(
                f"[Text {i}] (Score: {item['score']:.3f})\n"
                f"{item['content']}\n"
            )
    
    if context['tables']:
        context_parts.append("\n=== TABLE CONTENT ===\n")
        for i, item in enumerate(context['tables'], 1):
            context_parts.append(
                f"[Table {i}] {item['title']} (Score: {item['score']:.3f})\n"
                f"{item['content']}\n"
            )
    
    if context['images']:
        context_parts.append("\n=== IMAGES ===\n")
        for i, item in enumerate(context['images'], 1):
            context_parts.append(
                f"[Image {i}] (Score: {item['score']:.3f})\n"
                f"{item['content']}\n"
            )
    
    if context['audio']:
        context_parts.append("\n=== AUDIO TRANSCRIPTS ===\n")
        for i, item in enumerate(context['audio'], 1):
            context_parts.append(
                f"[Audio {i}] {item['speaker']} at {item['timestamp']} "
                f"(Score: {item['score']:.3f})\n"
                f"{item['content']}\n"
            )
    
    text_context = "\n".join(context_parts)
    image_paths = context['image_paths']
    
    return text_context, image_paths


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
