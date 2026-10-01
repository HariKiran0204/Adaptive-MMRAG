import os
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# ── OFFLINE LOCK (must be set BEFORE sentence_transformers / HuggingFace) ──
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer, CrossEncoder = None, None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

import re
import hashlib
from datetime import datetime


# --- Configuration ---
# Use standard public model IDs so cached downloads work offline.
EMBEDDING_MODEL = 'clip-ViT-B-32'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

_GLOBAL_EMBEDDING_GENERATOR = None
_GLOBAL_RERANKER = None

def get_unified_embedding_generator() -> 'UnifiedEmbeddingGenerator':
    """Return cached singleton instance of UnifiedEmbeddingGenerator to avoid repeated cold disk loads."""
    global _GLOBAL_EMBEDDING_GENERATOR
    if _GLOBAL_EMBEDDING_GENERATOR is None:
        _GLOBAL_EMBEDDING_GENERATOR = UnifiedEmbeddingGenerator()
    return _GLOBAL_EMBEDDING_GENERATOR

def get_reranker():
    """Return cached singleton instance of CrossEncoder reranker."""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None and CrossEncoder is not None:
        try:
            print(f"Loading cross-encoder reranker: {RERANKER_MODEL}...")
            _GLOBAL_RERANKER = CrossEncoder(RERANKER_MODEL, max_length=512, local_files_only=True)
            print("Reranker loaded successfully!")
        except Exception as e:
            print(f"⚠️  Reranker unavailable: {e}")
    return _GLOBAL_RERANKER


# ── Common visual vocabulary for zero-shot CLIP classification ─────────────
_CLIP_LABELS = [
    # Fruits & food
    "apple", "banana", "orange", "mango", "grapes", "strawberry", "watermelon",
    "pineapple", "lemon", "cherry", "pear", "peach", "avocado",
    "pizza", "burger", "sandwich", "cake", "bread", "rice", "salad",
    "vegetables", "carrot", "tomato", "potato",
    # Animals
    "dog", "cat", "bird", "fish", "horse", "cow", "elephant", "tiger", "lion",
    "rabbit", "monkey", "bear", "deer", "butterfly", "snake",
    # Vehicles
    "car", "motorcycle", "bicycle", "bus", "truck", "airplane", "ship", "train",
    # Nature & scenes
    "flower", "rose", "sunflower", "tree", "forest", "mountain", "beach", "ocean",
    "river", "lake", "sky", "clouds", "sunset", "garden", "grass",
    # Objects & places
    "house", "building", "bridge", "road", "city", "village",
    "computer", "laptop", "phone", "keyboard", "camera", "television",
    "book", "pen", "clock", "chair", "table", "sofa", "bed",
    # People & activities
    "person", "people", "face", "crowd", "baby", "child",
    "sports", "football", "cricket", "swimming", "running",
    # Documents & text content
    "document", "chart", "graph", "map", "diagram", "screenshot",
]


def classify_image_with_clip(image_path: str, top_k: int = 3) -> str:
    """
    Zero-shot image classification using the already-loaded CLIP (clip-ViT-B-32) singleton.
    Encodes the image and compares it to text labels using cosine similarity.
    100% offline — reuses the singleton model, no extra download needed.

    Returns a short description string like:
        "apple (confidence: high), fruit, food"
    or empty string "" if CLIP model is unavailable.
    """
    try:
        gen = get_unified_embedding_generator()
        if gen is None or gen.model is None:
            return ""

        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            img_rgb = img.convert('RGB')

        # Encode the image
        img_emb = gen.model.encode(img_rgb, convert_to_numpy=True, normalize_embeddings=True)

        # Encode all label texts as "a photo of X"
        texts = [f"a photo of {label}" for label in _CLIP_LABELS]
        text_embs = gen.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=False)

        # Cosine similarity (both are already L2-normalized so dot product = cosine sim)
        scores = text_embs @ img_emb  # shape: (num_labels,)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        top_results = [((_CLIP_LABELS[i], float(scores[i]))) for i in top_indices]

        if not top_results:
            return ""

        best_label, best_score = top_results[0]

        # Confidence tier: CLIP cosine scores typically 0.15–0.35 for correct matches
        if best_score >= 0.28:
            confidence = "high confidence"
        elif best_score >= 0.20:
            confidence = "moderate confidence"
        else:
            confidence = "low confidence"

        # Build description
        primary = best_label
        secondary = [lbl for lbl, _ in top_results[1:]]
        desc = f"{primary} ({confidence})"
        if secondary:
            desc += f"; possibly also: {', '.join(secondary)}"
        return desc

    except Exception as e:
        return ""



def extract_document_version_info(source_filename: str) -> Tuple[str, float]:
    """
    Extract document_id and version number from source filename.
    Example: 'company_policy_v2.pdf' -> ('company_policy', 2.0)
             'annual_report_v1.2.docx' -> ('annual_report', 1.2)
             'notes.txt' -> ('notes', 1.0)
    """
    if not source_filename or source_filename == 'unknown':
        return ('unknown_doc', 1.0)
    
    base_name = os.path.splitext(os.path.basename(source_filename))[0]
    
    # Check for version pattern like _v2, -v1.2, _version_3
    v_match = re.search(r'[_\-\s](?:v|version)[_\-\s]?(\d+(?:\.\d+)?)', base_name, re.IGNORECASE)
    if v_match:
        try:
            version = float(v_match.group(1))
            doc_id = re.sub(r'[_\-\s](?:v|version)[_\-\s]?(\d+(?:\.\d+)?)', '', base_name, flags=re.IGNORECASE).strip(' _-')
            return (doc_id or base_name, version)
        except ValueError:
            pass
    
    # Also check standalone v2 or v1 at end of name
    v_match2 = re.search(r'v(\d+(?:\.\d+)?)$', base_name, re.IGNORECASE)
    if v_match2:
        try:
            version = float(v_match2.group(1))
            doc_id = base_name[:v_match2.start()].strip(' _-')
            return (doc_id or base_name, version)
        except ValueError:
            pass
            
    return (base_name, 1.0)


class UnifiedEmbeddingGenerator:
    """Generates embeddings for all content types using a single CLIP model."""
    def __init__(self):
        """Initializes the unified CLIP model."""
        print(f"Loading unified CLIP embedding model: {EMBEDDING_MODEL}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device, local_files_only=True)
        print(f"Model loaded on device: {self.device}")

    def embed(self, content: Any) -> np.ndarray:
        """Embeds either text (str) or an image (PIL Image object)."""
        return self.model.encode(content, convert_to_numpy=True, normalize_embeddings=True)

    def process_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Processes chunks and generates embeddings with vectorized batching for maximum speed."""
        if not chunks:
            return {'embeddings': np.array([]), 'metadata': []}
            
        print(f"Generating unified CLIP embeddings for {len(chunks)} chunks (vectorized batching)...")
        embeddings = [None] * len(chunks)
        metadata = []
        
        # Separate images from texts for vectorized batch processing
        text_indices = []
        text_contents = []
        
        for i, chunk in enumerate(chunks):
            content_type = chunk.get('metadata', {}).get('type', 'text')
            is_image = (content_type == 'image' and 'image_path' in chunk.get('metadata', {}))
            
            if is_image:
                try:
                    img_p = chunk['metadata']['image_path']
                    if img_p and Path(img_p).is_file():
                        image = Image.open(img_p).convert('RGB')
                        embeddings[i] = self.embed(image)
                    else:
                        text_indices.append(i)
                        text_contents.append(str(chunk.get('content', '')))
                except Exception as e:
                    text_indices.append(i)
                    text_contents.append(str(chunk.get('content', '')))
            else:
                text_indices.append(i)
                text_contents.append(str(chunk.get('content', '')))
                
        # Vectorized batch encoding for all text chunks
        if text_contents:
            batch_embs = self.model.encode(
                text_contents, 
                batch_size=32, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            for idx, emb in zip(text_indices, batch_embs):
                embeddings[idx] = emb
                
        # Build rich versioning & context metadata in exact order
        for i, chunk in enumerate(chunks):
            content_type = chunk.get('metadata', {}).get('type', 'text')
            source_name = chunk.get('metadata', {}).get('source', 'unknown')
            doc_id, detected_ver = extract_document_version_info(source_name)
            final_doc_id = chunk.get('metadata', {}).get('document_id', doc_id)
            final_version = float(chunk.get('metadata', {}).get('version', detected_ver))
            content_str = str(chunk.get('content', ''))
            content_hash = hashlib.sha256(content_str.encode('utf-8', errors='ignore')).hexdigest()[:16]
            ts = chunk.get('metadata', {}).get('timestamp', datetime.now().isoformat())

            metadata.append({
                'chunk_id': chunk.get('id', f'chunk_{i}'),
                'document_id': final_doc_id,
                'version': final_version,
                'content_hash': content_hash,
                'timestamp': ts,
                'type': content_type,
                'content': content_str,
                'content_preview': content_str[:200] + '...' if len(content_str) > 200 else content_str,
                'source': source_name,
                'page_number': chunk.get('metadata', {}).get('page_number', 'N/A'),
                'chunk_index': chunk.get('metadata', {}).get('chunk_index', 0),
                'total_chunks': chunk.get('metadata', {}).get('total_chunks', 1),
                'token_count': chunk.get('token_count', 0),
                'original_metadata': chunk.get('metadata', {})
            })

        valid_embeddings = [emb for emb in embeddings if emb is not None]
        return {
            'embeddings': np.array(valid_embeddings),
            'metadata': metadata
        }


class BM25HybridSearchEngine:
    """
    Advanced hybrid search combining:
    1. Semantic search (FAISS + CLIP)
    2. BM25 keyword search
    3. Cross-encoder reranking
    """
    
    def __init__(self, index, metadata: List[Dict], generator: UnifiedEmbeddingGenerator):
        self.index = index
        self.metadata = metadata
        self.generator = generator
        
        # Initialize BM25
        print("Building BM25 index...")
        self.bm25 = self._build_bm25_index()
        
        # Initialize cross-encoder reranker using singleton cache
        self.reranker = get_reranker()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split and filter out short tokens
        tokens = [token for token in text.split() if len(token) > 2]
        return tokens
    
    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index from document corpus."""
        # Tokenize all documents
        corpus_tokens = []
        for meta in tqdm(self.metadata, desc="Tokenizing corpus for BM25"):
            content = meta['content']
            tokens = self._tokenize(content)
            corpus_tokens.append(tokens)
        
        # Create BM25 index
        bm25 = BM25Okapi(corpus_tokens)
        print(f"BM25 index built with {len(corpus_tokens)} documents")
        return bm25
    
    def bm25_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform BM25 search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, bm25_score) tuples
        """
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # Normalize scores to [0, 1] range
        max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                normalized_score = scores[idx] / max_score
                results.append((int(idx), float(normalized_score)))
        
        return results
    
    def semantic_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform semantic search using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        query_embedding = self.generator.embed(query)
        if query_embedding.ndim == 1:
            query_embedding = np.array([query_embedding])
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append((int(idx), float(similarities[0][i])))
        return results
    
    def reciprocal_rank_fusion(self, 
                               search_results_list: List[List[Tuple[int, float]]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
        
        RRF is a simple but effective method that doesn't require score normalization.
        Score for document d = sum over all rankings of 1 / (k + rank(d))
        
        Args:
            search_results_list: List of search results from different methods
            k: Constant for RRF formula (typically 60)
            
        Returns:
            Fused ranked list of (document_index, fused_score) tuples
        """
        # Calculate RRF scores
        rrf_scores = {}
        
        for search_results in search_results_list:
            for rank, (doc_id, _) in enumerate(search_results, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def rerank_with_cross_encoder(self, 
                                   query: str, 
                                   candidates: List[Tuple[int, float]], 
                                   top_k: int = 15) -> List[Dict]:
        """
        Rerank candidates using cross-encoder for final ranking.
        
        Args:
            query: Search query
            candidates: List of (document_index, score) tuples
            top_k: Number of final results to return
            
        Returns:
            List of reranked results with metadata
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        candidate_indices = []
        
        for doc_idx, _ in candidates[:50]:  # Rerank top 50 candidates
            content = self.metadata[doc_idx]['content']
            # Truncate very long content for cross-encoder
            content_truncated = content[:1000] if len(content) > 1000 else content
            pairs.append([query, content_truncated])
            candidate_indices.append(doc_idx)
        
        # Get cross-encoder scores
        print(f"Reranking {len(pairs)} candidates with cross-encoder...")
        ce_scores = self.reranker.predict(pairs)
        
        # Combine with original scores and create final ranking
        reranked_results = []
        for idx, ce_score in zip(candidate_indices, ce_scores):
            reranked_results.append({
                'doc_idx': idx,
                'ce_score': float(ce_score),
                'metadata': self.metadata[idx]
            })
        
        # Sort by cross-encoder score
        reranked_results.sort(key=lambda x: x['ce_score'], reverse=True)
        
        # Format final results
        final_results = []
        for i, result in enumerate(reranked_results[:top_k]):
            final_results.append({
                "score": result['ce_score'],
                "rank": i + 1,
                "metadata": result['metadata']
            })
        
        return final_results

    def prioritize_version_aware_candidates(
        self, 
        candidates: List[Tuple[int, float]], 
        query: str
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """
        Prioritizes the highest version number of a document unless the query
        explicitly asks about an older version by name (e.g. 'v1', 'version 1', 'old').
        
        Returns:
            (reordered_candidates, version_rerank_applied)
        """
        if not candidates or len(candidates) <= 1:
            return candidates, False

        q_lower = query.lower()
        # Check if query specifically requests an older version
        asks_older = bool(re.search(r'\b(v1|v1\.0|version\s*1|old|previous|earlier|deprecated)\b', q_lower))
        if asks_older:
            return candidates, False

        # Group candidate document indices by document_id and find max version per document
        max_version_per_doc = {}
        for doc_idx, score in candidates:
            meta = self.metadata[doc_idx]
            doc_id = meta.get('document_id', 'unknown')
            ver = float(meta.get('version', 1.0))
            if doc_id not in max_version_per_doc or ver > max_version_per_doc[doc_id]:
                max_version_per_doc[doc_id] = ver

        # Boost scores for chunks from the highest version, apply slight penalty for outdated versions
        adjusted_candidates = []
        applied = False
        
        for doc_idx, score in candidates:
            meta = self.metadata[doc_idx]
            doc_id = meta.get('document_id', 'unknown')
            ver = float(meta.get('version', 1.0))
            max_ver = max_version_per_doc.get(doc_id, 1.0)
            
            if ver < max_ver:
                # Outdated version in presence of newer version: downweight
                new_score = score * 0.70
                applied = True
            elif ver == max_ver and max_ver > 1.0:
                # Latest version: boost priority
                new_score = score * 1.25
                applied = True
            else:
                new_score = score
                
            adjusted_candidates.append((doc_idx, new_score))

        # Re-sort if any adjustments were made
        if applied:
            adjusted_candidates.sort(key=lambda x: x[1], reverse=True)
            orig_indices = [c[0] for c in candidates]
            new_indices = [c[0] for c in adjusted_candidates]
            applied = (orig_indices != new_indices)

        return adjusted_candidates, applied
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 15,
                     semantic_weight: float = 0.5,
                     bm25_weight: float = 0.5,
                     use_reranker: bool = True) -> List[Dict]:
        """
        Perform hybrid search combining semantic + BM25 with optional reranking.
        
        Pipeline:
        1. Retrieve candidates from semantic search (top 50)
        2. Retrieve candidates from BM25 (top 50)
        3. Fuse results using Reciprocal Rank Fusion
        4. Version-Aware Document Prioritization
        5. Rerank top candidates with cross-encoder
        
        Args:
            query: Search query
            k: Number of final results
            semantic_weight: Weight for semantic search (not used in RRF)
            bm25_weight: Weight for BM25 (not used in RRF)
            use_reranker: Whether to use cross-encoder reranking
            
        Returns:
            List of top k results with scores and metadata
        """
        print(f"\n🔍 Hybrid Search Pipeline:")
        print(f"   Query: '{query}'")
        
        # Step 1: Semantic search
        print(f"   1️⃣ Semantic search (retrieving top 50)...")
        semantic_results = self.semantic_search(query, k=50)
        print(f"      ✓ Retrieved {len(semantic_results)} semantic results")
        
        # Step 2: BM25 search
        print(f"   2️⃣ BM25 keyword search (retrieving top 50)...")
        bm25_results = self.bm25_search(query, k=50)
        print(f"      ✓ Retrieved {len(bm25_results)} BM25 results")
        
        # Step 3: Reciprocal Rank Fusion
        print(f"   3️⃣ Fusing results with Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            k=60
        )
        print(f"      ✓ Fused to {len(fused_results)} unique documents")
        
        # Step 3.5: Version-aware document prioritization
        fused_results, version_applied = self.prioritize_version_aware_candidates(fused_results, query)
        if version_applied:
            print(f"      🗂️ Version-aware prioritization applied: newest document versions prioritized")
        
        # Step 4: Cross-encoder reranking
        if use_reranker and self.reranker is not None:
            print(f"   4️⃣ Reranking with cross-encoder...")
            final_results = self.rerank_with_cross_encoder(query, fused_results, top_k=k)
            print(f"      ✓ Reranked to top {len(final_results)} results")
        else:
            # Skip reranking, just format results
            final_results = []
            for i, (doc_idx, score) in enumerate(fused_results[:k]):
                final_results.append({
                    "score": score,
                    "rank": i + 1,
                    "metadata": self.metadata[doc_idx]
                })
        
        print(f"   ✅ Final: {len(final_results)} results ready\n")
        return final_results


def create_and_save_faiss_index(embeddings: np.ndarray, metadata: list, 
                                output_path: str = "faiss_index.bin"):
    """Creates and saves a FAISS index optimized for cosine similarity."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    print(f"Adding {embeddings.shape[0]} vectors to the FAISS index...")
    index.add(embeddings)

    faiss.write_index(index, output_path)
    
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    
    # Quick fix: Convert booleans when saving
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.integer)):
                return int(obj)
            elif isinstance(obj, (np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

    print(f"âœ… FAISS index saved to '{output_path}'")
    print(f"âœ… Metadata saved to '{metadata_path}'")

def load_faiss_index(index_path: str = "faiss_index.bin"):
    """Loads a FAISS index and its corresponding metadata."""
    metadata_path = os.path.splitext(index_path)[0] + "_metadata.json"
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"FAISS index or metadata not found.")

    print(f"Loading FAISS index from '{index_path}'...")
    index = faiss.read_index(index_path)
    
    print(f"Loading metadata from '{metadata_path}'...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return index, metadata


def main():
    """Main function to run the full indexing and testing pipeline."""
    chunks_file_path = "/Users/vijay/Documents/SIH/audio_chunks.pkl"
    try:
        print(f"Loading raw chunks from: {chunks_file_path}")
        with open(chunks_file_path, 'rb') as f:
            raw_chunks = pickle.load(f)
        print(f"Successfully loaded {len(raw_chunks)} chunks.")
    except FileNotFoundError:
        print(f"Error: The file '{chunks_file_path}' was not found.")
        return

    # Generate embeddings with full content
    generator = UnifiedEmbeddingGenerator()
    embeddings_data = generator.process_chunks(raw_chunks)
    
    embeddings = embeddings_data['embeddings']
    metadata = embeddings_data['metadata']

    # Create and save the FAISS index
    faiss_file_path = "document_index.bin"
    create_and_save_faiss_index(embeddings, metadata, faiss_file_path)

    # Load and test BM25 hybrid search with reranking
    try:
        loaded_index, loaded_metadata = load_faiss_index(faiss_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Initialize BM25 hybrid search engine with reranker
    print("\n" + "="*80)
    print("Initializing BM25 Hybrid Search Engine with Cross-Encoder Reranker")
    print("="*80)
    hybrid_engine = BM25HybridSearchEngine(loaded_index, loaded_metadata, generator)
    
    # Test query
    query = "explain about the the contents in the complex tables"
    print(f"\n{'='*80}")
    print(f"Testing: '{query}'")
    print("="*80)
    
    search_results = hybrid_engine.hybrid_search(query, k=10, use_reranker=True)

    print(f"\n{'='*80}")
    print(f"TOP 10 RERANKED RESULTS")
    print("="*80)
    for i, result in enumerate(search_results, 1):
        meta = result['metadata']
        print(f"\n{i}. Rank: {result['rank']} | Cross-Encoder Score: {result['score']:.4f}")
        print(f"   Type: {meta['type']}")
        print(f"   Source: {meta['source']} (Page: {meta.get('page_number', 'N/A')})")
        print(f"   Preview: {meta['content_preview']}")

if __name__ == "__main__":
    main()
