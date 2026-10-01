"""
nlp_summarizer.py - NLP Document Summarization & Keyphrase Extraction Engine
for Adaptive-MMRAG.

Features:
1. Pure offline Keyphrase Extraction (salience, position-weighted TF-IDF, collocations).
2. Extractive Summarization (Sentence centrality & salience scoring).
3. Local Abstractive Summarization via Ollama (qwen2.5:3b) with graceful fallback.
4. Per-document profile generation & persistent caching.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False


class DocumentNLPSummarizer:
    """
    Offline NLP Summarizer & Keyphrase Extractor for ingested multimodal documents.
    """

    STOPWORDS = {
        'the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it', 'with', 'as',
        'his', 'on', 'be', 'at', 'by', 'i', 'this', 'had', 'not', 'are', 'but', 'from', 'or', 'have',
        'an', 'they', 'which', 'one', 'you', 'were', 'her', 'all', 'she', 'there', 'would', 'their',
        'we', 'him', 'been', 'has', 'when', 'who', 'will', 'more', 'no', 'if', 'out', 'so', 'said',
        'what', 'up', 'its', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new', 'some',
        'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first', 'any', 'my', 'now', 'such',
        'like', 'our', 'over', 'man', 'me', 'even', 'most', 'made', 'after', 'also', 'did', 'many',
        'before', 'must', 'through', 'back', 'years', 'where', 'much', 'your', 'way', 'well', 'down',
        'should', 'because', 'each', 'just', 'those', 'people', 'how', 'too', 'little', 'state', 'good',
        'very', 'make', 'world', 'still', 'own', 'see', 'men', 'work', 'long', 'get', 'here', 'between',
        'both', 'life', 'being', 'under', 'never', 'day', 'same', 'another', 'know', 'while', 'last',
        'might', 'great', 'us', 'year', 'off', 'come', 'since', 'against', 'go', 'came', 'right', 'used',
        'take', 'three', 'himself', 'look', 'few', 'during', 'without', 'again', 'around', 'however',
        'found', 'part', 'once', 'general', 'show', 'case', 'page', 'source', 'chunk', 'document',
        'cid', 'email', 'phone', 'http', 'https', 'www', 'com', 'gmail', 'linkedin', 'github'
    }

    def __init__(self, cache_file: str = "processed_output/document_nlp_profiles.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(exist_ok=True)
        self.profiles: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cached profiles if available."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persist profiles to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[NLPSummarizer][WARN] Failed to save profile cache: {e}")

    # Image file extensions — for these, we show filename as the tag, not headings
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg'}

    def extract_headings(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract document section headings as display tags.

        A heading line is:
        - Short: 1–8 words
        - Starts with a capital letter OR a numbered prefix like "1.", "2.1", "Chapter 3", "Section A"
        - Does NOT end with a period, comma, semicolon (those are sentences)
        - At least 3 characters long
        - Not a pure number or single word stopword
        """
        if not text or len(text.strip()) < 10:
            return []

        headings = []
        seen = set()

        # Split into lines — headings are line-level elements
        lines = re.split(r'\n+|\r\n+', text)

        # Pattern: numbered prefix like "1.", "1.1", "2.3.4", "Chapter 3", "Section A"
        numbered_prefix = re.compile(
            r'^(\d+(\.\d+)*\.?\s+|Chapter\s+\d+|Section\s+[\dA-Z])',
            re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            words = line.split()
            if len(words) < 1 or len(words) > 8:
                continue  # too short or too long to be a heading

            # Must not end with sentence-ending punctuation
            if line.endswith(('.', ',', ';', ':', '?')):
                continue

            # Must start with capital OR numbered prefix
            is_capitalized = line[0].isupper()
            is_numbered = bool(numbered_prefix.match(line))

            if not (is_capitalized or is_numbered):
                continue

            # Reject pure single stopwords
            if len(words) == 1 and words[0].lower() in self.STOPWORDS:
                continue

            # Deduplicate case-insensitively
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)

            headings.append(line)
            if len(headings) >= top_k:
                break

        return headings

    def extract_keyphrases(self, text: str, top_k: int = 6) -> List[str]:

        """
        Extract salient domain keyphrases using position-weighted frequency,
        capitalized entity detection, and meaningful collocations.
        """
        if not text or len(text.strip()) < 10:
            return []

        # Find individual meaningful words
        raw_words = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text)
        if not raw_words:
            return []

        total_words = len(raw_words)
        word_scores = Counter()

        # Score words with position weight (earlier words weighted up to 1.5x)
        for idx, w in enumerate(raw_words):
            lw = w.lower()
            if lw not in self.STOPWORDS and not lw.isdigit():
                pos_weight = 1.0 + 0.5 * (1.0 - (idx / max(total_words, 1)))
                word_scores[lw] += pos_weight

        # Extract 2-word and 3-word noun-phrase candidates
        phrase_scores = Counter()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        for s in sentences:
            tokens = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s)
            for n in (2, 3):
                for i in range(len(tokens) - n + 1):
                    gram = tokens[i:i+n]
                    # Filter if starts or ends with stopword
                    if gram[0].lower() in self.STOPWORDS or gram[-1].lower() in self.STOPWORDS:
                        continue
                    if any(w.lower() in self.STOPWORDS for w in gram):
                        continue
                    phrase = " ".join(gram).lower()
                    score = sum(word_scores.get(w.lower(), 1.0) for w in gram) * (1.2 if n == 2 else 1.4)
                    phrase_scores[phrase] += score

        candidates = []
        # Add top multi-word phrases
        for phrase, _ in phrase_scores.most_common(top_k * 2):
            if len(phrase) >= 5 and not any(phrase in c for c in candidates):
                candidates.append(phrase)

        # Fill with top single keywords if needed
        for word, _ in word_scores.most_common(top_k * 2):
            if len(candidates) >= top_k:
                break
            if not any(word in c for c in candidates):
                candidates.append(word)

        return candidates[:top_k]

    def extract_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Extractive summarization using sentence centrality and keyphrase salience.
        Executes in <15ms offline.
        """
        if not text or len(text.strip()) < 30:
            return text.strip()

        # Split into sentences
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if len(s.strip()) >= 20]
        if not raw_sentences:
            return text[:250].strip() + "..."

        if len(raw_sentences) <= num_sentences:
            return " ".join(raw_sentences)

        keyphrases = set(self.extract_keyphrases(text, top_k=10))

        # Score sentences
        sentence_scores = []
        for idx, sentence in enumerate(raw_sentences):
            words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', sentence.lower()))
            if not words:
                continue

            # 1. Salience: Keyphrase overlap
            overlap = sum(1 for kp in keyphrases if any(k_part in words for k_part in kp.split()))
            # 2. Position bias: lead sentences carry higher weight
            pos_weight = 1.4 if idx == 0 else (1.2 if idx < 3 else 1.0)
            # 3. Length normalization (penalize extremely short or long sentences)
            length_norm = 1.0 if 40 <= len(sentence) <= 220 else 0.7

            score = (overlap + 0.5) * pos_weight * length_norm
            sentence_scores.append((score, idx, sentence))

        # Select top sentences by score, but restore chronological order
        top_selected = sorted(sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences], key=lambda x: x[1])
        summary = " ".join([s[2] for s in top_selected])
        return summary

    def generate_abstractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        Generates an abstractive executive summary via local Ollama LLM.
        Falls back to extractive summary if Ollama is not active or times out.
        """
        if not OLLAMA_AVAILABLE:
            return self.extract_summary(text, num_sentences=max_sentences)

        lead_text = text[:1500].strip()
        try:
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an NLP executive document summarizer. "
                            f"Write a clear, concise {max_sentences}-sentence executive summary of the document. "
                            "Focus strictly on key objectives, findings, and figures. Do not write any conversational preamble."
                        )
                    },
                    {"role": "user", "content": f"Document text:\n{lead_text}"}
                ],
                options={"temperature": 0.1, "num_predict": 120}
            )
            result = (response.get("message", {}).get("content") or "").strip()
            if len(result) > 30:
                return result
        except Exception as e:
            print(f"[NLPSummarizer][INFO] Local LLM summary fallback to extractive: {e}")

        return self.extract_summary(text, num_sentences=max_sentences)

    def summarize_document(
        self,
        text: str,
        filename: str,
        chunk_count: int = 1,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete NLP profile for a document:
        - Keyphrases (tags)
        - Executive Summary
        - Metadata (word count, chunk count)
        """
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        word_count = len(cleaned_text.split())

        # Check cache first
        cached = self.profiles.get(filename)
        if cached and cached.get("word_count") == word_count:
            return cached

        # ── Tag extraction: prefer document headings over word frequencies ──
        # Try headings first (raw text before whitespace-collapse to preserve line structure)
        headings = self.extract_headings(text, top_k=5)
        if len(headings) >= 2:
            keyphrases = headings          # Use actual section names as tags
        else:
            keyphrases = self.extract_keyphrases(cleaned_text, top_k=5)  # fallback

        if use_llm and OLLAMA_AVAILABLE and word_count >= 40:
            summary = self.generate_abstractive_summary(cleaned_text)
        else:
            summary = self.extract_summary(cleaned_text, num_sentences=3)

        profile = {
            "filename": filename,
            "keyphrases": keyphrases,
            "summary": summary,
            "word_count": word_count,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        self.profiles[filename] = profile
        self._save_cache()
        return profile

    def process_corpus(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group corpus chunks by document source and generate summary profiles.
        For image files: tag is just the image filename (no text to extract headings from).
        For text/pdf/docx: extract headings as tags.
        """
        docs_text: Dict[str, List[str]] = {}
        docs_count: Dict[str, int] = {}
        docs_raw_text: Dict[str, List[str]] = {}  # raw (un-joined) lines for heading extraction

        for meta in metadata_list:
            source = (
                meta.get("source")
                or (meta.get("metadata", {}).get("source") if isinstance(meta.get("metadata"), dict) else None)
                or (meta.get("original_metadata", {}).get("source") if isinstance(meta.get("original_metadata"), dict) else None)
                or "unknown_doc"
            )
            base_source = os.path.basename(source)
            content = (
                meta.get("content")
                or meta.get("text")
                or (meta.get("metadata", {}).get("content") if isinstance(meta.get("metadata"), dict) else "")
                or ""
            )
            if base_source not in docs_text:
                docs_text[base_source] = []
                docs_raw_text[base_source] = []
                docs_count[base_source] = 0
            if content.strip():
                docs_text[base_source].append(content.strip())
                # Keep raw content with newlines for heading detection
                raw = meta.get("content") or meta.get("text") or ""
                docs_raw_text[base_source].append(raw)
                docs_count[base_source] += 1

        print(f"[NLPSummarizer] Generating NLP profiles for {len(docs_text)} document(s)...")
        results = {}
        for source, chunks in docs_text.items():
            file_ext = Path(source).suffix.lower()

            # Image files — use filename as the sole tag
            if file_ext in self.IMAGE_EXTS:
                stem = Path(source).stem  # filename without extension
                profile = {
                    "filename": source,
                    "keyphrases": [stem],
                    "summary": f"Indexed image: {source}",
                    "word_count": 0,
                    "chunk_count": docs_count.get(source, 1),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "is_image": True
                }
                self.profiles[source] = profile
                results[source] = profile
                continue

            # Text/PDF/DOCX — extract headings from raw line-preserving text
            raw_full = "\n".join(docs_raw_text[source])
            joined_text = " ".join(chunks)
            profile = self.summarize_document(
                text=raw_full or joined_text,   # prefer raw (has line breaks for headings)
                filename=source,
                chunk_count=docs_count.get(source, 1),
                use_llm=False
            )
            results[source] = profile

        return results


    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        return self.profiles

    def clear_cache(self):
        self.profiles = {}
        if self.cache_file.exists():
            try:
                os.remove(self.cache_file)
            except Exception:
                pass

# Global singleton
nlp_summarizer = DocumentNLPSummarizer()

def process_corpus(metadata_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return nlp_summarizer.process_corpus(metadata_list)

def get_all_summaries() -> Dict[str, Dict[str, Any]]:
    return nlp_summarizer.get_all_summaries()

def clear_profiles() -> None:
    nlp_summarizer.clear_cache()

def summarize_document(text: str, filename: str, chunk_count: int = 1, use_llm: bool = False) -> Dict[str, Any]:
    return nlp_summarizer.summarize_document(text, filename, chunk_count, use_llm)

