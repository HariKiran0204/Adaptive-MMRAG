
# image_processor.py - Complete Image Processing System for Multimodal RAG

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# ── OFFLINE LOCK (must be set BEFORE importing transformers / HuggingFace) ──
# Prevents any attempt to phone home to huggingface.co or cdn.huggingface.co
os.environ.setdefault('HF_HUB_OFFLINE', '1')          # Never ping HuggingFace Hub
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')     # Use cached model files only
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')      # Datasets: local cache only
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Silence tokenizer warning

import torch
from transformers import CLIPProcessor, CLIPModel
import re
from collections import Counter

# Configure local tessdata path
project_root = os.path.dirname(os.path.abspath(__file__))
local_tessdata = os.path.join(project_root, "tessdata")
if os.path.exists(local_tessdata):
    os.environ["TESSDATA_PREFIX"] = local_tessdata

def get_tesseract_languages() -> str:
    """Detect available language packs in the local or system Tesseract folder and return combined string (e.g., 'eng+tel')."""
    try:
        langs = ["eng"]
        tessdata_path = os.environ.get("TESSDATA_PREFIX")
        if not tessdata_path or not os.path.exists(tessdata_path):
            tessdata_path = r"C:\Program Files\Tesseract-OCR\tessdata"
            
        if os.path.exists(tessdata_path):
            for f in os.listdir(tessdata_path):
                if f.endswith(".traineddata"):
                    lang_code = f.replace(".traineddata", "")
                    if lang_code not in ["osd", "eng"] and len(lang_code) == 3:
                        langs.append(lang_code)
        return "+".join(langs)
    except Exception:
        return "eng"

@dataclass
class ImageProcessingConfig:
    """Configuration for image processing."""
    embedding_dim: int = 512
    ocr_confidence_threshold: float = 50.0
    min_words_for_text: int = 10
    upscale_threshold: int = 1000  # pixels
    max_file_size_mb: float = 50.0
    screenshot_keywords: List[str] = None
    
    def __post_init__(self):
        if self.screenshot_keywords is None:
            self.screenshot_keywords = [
                'screenshot', 'screen shot', 'window', 'browser',
                'chrome', 'safari', 'firefox', 'app', 'application',
                'menu', 'file', 'edit', 'view', 'settings', 'gmail',
                'slack', 'zoom', 'teams', 'outlook', 'search', 'inbox'
            ]


class ImagePreprocessor:
    """Handles image preprocessing for better OCR."""
    
    @staticmethod
    def create_variants(image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """
        Create industrial-grade deblurred and contrast-enhanced versions for OCR.
        
        Returns:
            List of (variant_name, processed_image) tuples
        """
        variants = []
        
        # 1. Original
        variants.append(("original", image.copy()))
        
        try:
            import cv2
            img_np = np.array(image)
            if len(img_np.shape) == 3:
                gray_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_cv = img_np.copy()
            
            # Super-resolution / Upscaling for small images (makes fine fonts crystal clear)
            h, w = gray_cv.shape
            if w < 1600 or h < 1600:
                scale = max(1600 / w, 1600 / h)
                gray_cv = cv2.resize(gray_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            
            # 2. Kernel Deblurring using Unsharp Masking (removes camera blur & motion blur)
            gaussian = cv2.GaussianBlur(gray_cv, (0, 0), 2.0)
            deblurred = cv2.addWeighted(gray_cv, 1.6, gaussian, -0.6, 0)
            variants.append(("deblurred_sharp", Image.fromarray(deblurred)))
            
            # 3. Morphological Background Division (Eliminates security watermarks & diagonal seals on Govt certificates)
            dilated = cv2.morphologyEx(deblurred, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))
            bg_clean = cv2.divide(deblurred, dilated, scale=255)
            bg_clean_thresh = cv2.threshold(bg_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            variants.append(("watermark_removed", Image.fromarray(bg_clean_thresh)))
            
            # 4. CLAHE (Local Adaptive Contrast) - handles uneven phone lighting and shadows
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            clahe_enhanced = clahe.apply(deblurred)
            variants.append(("clahe_enhanced", Image.fromarray(clahe_enhanced)))
            
            # 5. Bilateral Denoised Otsu Binarization (Best for holographic ID cards and printed docs)
            denoised = cv2.bilateralFilter(clahe_enhanced, 7, 50, 50)
            thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            variants.append(("otsu_binarized", Image.fromarray(thresh)))
            
        except Exception as e:
            # Fallback PIL methods if OpenCV fails
            try:
                gray = image.convert('L')
                enhancer = ImageEnhance.Contrast(gray)
                variants.append(("high_contrast_gray", enhancer.enhance(2.0)))
                variants.append(("sharpened", image.filter(ImageFilter.SHARPEN)))
            except Exception:
                pass
        
        return variants


class OCREngine:
    """Advanced OCR with multiple strategies."""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.langs = get_tesseract_languages()
        print(f"      ℹ️  OCREngine initialized with languages: {self.langs}")
    
    def extract_text(self, image_path: str) -> Dict[str, any]:
        """
        Extract text using multiple OCR strategies.
        
        Returns:
            {
                'text': str,
                'confidence': float,
                'word_count': int,
                'has_text': bool,
                'words': List[Dict],
                'method': str,
                'all_attempts': List[Dict]
            }
        """
        try:
            image = Image.open(image_path)
            variants = self.preprocessor.create_variants(image)
            
            all_attempts = []
            best_result = None
            best_score = -1
            
            # Test English first to prevent regional language pack corruption on Latin ID cards
            lang_candidates = ["eng"]
            if self.langs and self.langs != "eng" and self.langs not in lang_candidates:
                lang_candidates.append(self.langs)
            
            # Optimized PSM modes for document and image reading
            psm_modes = [
                (6, "uniform_block"),  # Assume uniform block of text
                (3, "auto"),           # Fully automatic page segmentation
            ]
            
            for variant_name, variant_image in variants:
                for lang in lang_candidates:
                    for psm, psm_name in psm_modes:
                        try:
                            config_str = f'--psm {psm}'
                            ocr_data = pytesseract.image_to_data(
                                variant_image,
                                output_type=pytesseract.Output.DICT,
                                config=config_str,
                                lang=lang
                            )
                            
                            words_with_confidence = []
                            texts = []
                            confidences = []
                            
                            for i, text in enumerate(ocr_data['text']):
                                text_clean = text.strip()
                                if text_clean:
                                    try:
                                        conf = int(ocr_data['conf'][i])
                                        if conf > 0:
                                            texts.append(text_clean)
                                            confidences.append(conf)
                                            words_with_confidence.append({
                                                'word': text_clean,
                                                'confidence': conf,
                                                'bbox': (
                                                    ocr_data['left'][i],
                                                    ocr_data['top'][i],
                                                    ocr_data['width'][i],
                                                    ocr_data['height'][i]
                                                )
                                            })
                                    except (ValueError, IndexError):
                                        continue
                            
                            if confidences:
                                avg_confidence = float(np.mean(confidences))
                                full_text = ' '.join(texts)
                                
                                # Composite score balances both high confidence and meaningful text length
                                quality_score = avg_confidence * (len(texts) ** 0.5)
                                
                                result = {
                                    'text': full_text,
                                    'confidence': avg_confidence,
                                    'word_count': len(texts),
                                    'words': words_with_confidence,
                                    'method': f"{variant_name}_{lang}_{psm_name}",
                                    'variant': variant_name,
                                    'lang': lang,
                                    'psm': psm
                                }
                                
                                all_attempts.append(result)
                                
                                if quality_score > best_score:
                                    best_score = quality_score
                                    best_result = result
                                    
                                # Early exit if we have found clear, high-confidence text
                                if avg_confidence >= 80.0 and len(texts) >= 15:
                                    break
                        except Exception as e:
                            continue
                    
                    if best_score > 0 and best_result and best_result['confidence'] >= 85.0 and best_result['word_count'] >= 20:
                        break
                if best_score > 0 and best_result and best_result['confidence'] >= 85.0 and best_result['word_count'] >= 20:
                    break
            
            # Return best result
            if best_result:
                # Merge unique high-confidence tokens from other passes to ensure no ID numbers or details are missed
                primary_words = set(w.lower() for w in best_result['text'].split())
                extra_tokens = []
                for attempt in all_attempts:
                    if attempt != best_result and attempt.get('confidence', 0) >= 50.0:
                        for word in attempt.get('text', '').split():
                            w_clean = word.strip().strip(",.:;-")
                            if w_clean and len(w_clean) >= 3 and w_clean.lower() not in primary_words and any(c.isalnum() for c in w_clean):
                                extra_tokens.append(w_clean)
                                primary_words.add(w_clean.lower())
                
                if extra_tokens:
                    best_result['text'] = best_result['text'] + "\nAdditional Extracted Data: " + " ".join(extra_tokens)
                
                print(f"      ✓ Best OCR: {best_result['method']} ({best_result['confidence']:.1f}%)")
                best_result['has_text'] = (
                    len(best_result['text'].strip()) > 5 and 
                    best_result['confidence'] > self.config.ocr_confidence_threshold
                )
                best_result['all_attempts'] = all_attempts
                return best_result
            else:
                return self._empty_result()
        
        except Exception as e:
            print(f"⚠️  OCR completely failed: {e}")
            return self._empty_result(error=str(e))
    
    def _empty_result(self, error: str = None) -> Dict:
        """Return empty OCR result."""
        return {
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'has_text': False,
            'words': [],
            'method': 'none',
            'all_attempts': [],
            'error': error
        }


class ScreenshotDetector:
    """Detects if an image is a screenshot."""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
    
    def detect(self, image_path: str, ocr_result: Dict) -> Dict[str, any]:
        """
        Detect if image is a screenshot.
        
        Returns:
            {
                'is_screenshot': bool,
                'confidence': float,
                'indicators': dict
            }
        """
        try:
            image = Image.open(image_path)
            width, height = image.size
            aspect_ratio = width / height if height > 0 else 0
            
            ocr_text_lower = ocr_result['text'].lower()
            
            # Indicator 1: UI Keywords
            keyword_score = 0
            detected_keywords = []
            for keyword in self.config.screenshot_keywords:
                if keyword in ocr_text_lower:
                    keyword_score += 1
                    detected_keywords.append(keyword)
            
            # Indicator 2: Text density
            text_density = ocr_result['word_count'] / (width * height / 10000) if width * height > 0 else 0
            
            # Indicator 3: Aspect ratio
            common_ratios = [16/9, 16/10, 4/3, 3/2, 21/9]
            is_common_ratio = any(abs(aspect_ratio - ratio) < 0.1 for ratio in common_ratios)
            
            # Indicator 4: Timestamps
            timestamp_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'
            timestamps = re.findall(timestamp_pattern, ocr_result['text'])
            
            # Indicator 5: Application names
            app_names = []
            common_apps = ['chrome', 'safari', 'firefox', 'gmail', 'slack', 'zoom', 'teams', 'outlook', 'excel', 'word']
            for app in common_apps:
                if app in ocr_text_lower:
                    app_names.append(app.title())
            
            # Indicator 6: UI patterns
            ui_patterns = ['@', 'http', 'www', '.com', 'search', 'menu', 'settings']
            ui_score = sum(1 for pattern in ui_patterns if pattern in ocr_text_lower)
            
            # Decision logic
            is_screenshot = (
                keyword_score >= 2 or
                (text_density > 0.5 and is_common_ratio) or
                len(timestamps) > 0 or
                len(app_names) > 0 or
                ui_score >= 3
            )
            
            # Calculate confidence
            confidence = 0
            if keyword_score > 0:
                confidence += min(keyword_score * 0.15, 0.4)
            if is_common_ratio:
                confidence += 0.2
            if text_density > 0.5:
                confidence += 0.15
            if timestamps:
                confidence += 0.1
            if app_names:
                confidence += 0.1
            if ui_score >= 2:
                confidence += 0.05
            confidence = min(confidence, 1.0)
            
            return {
                'is_screenshot': is_screenshot,
                'confidence': float(confidence),
                'indicators': {
                    'keywords': detected_keywords,
                    'keyword_count': keyword_score,
                    'text_density': float(text_density),
                    'aspect_ratio': float(aspect_ratio),
                    'is_common_ratio': is_common_ratio,
                    'timestamps': timestamps,
                    'applications': app_names,
                    'ui_patterns': ui_score
                }
            }
        
        except Exception as e:
            print(f"⚠️  Screenshot detection failed: {e}")
            return {
                'is_screenshot': False,
                'confidence': 0,
                'indicators': {},
                'error': str(e)
            }


class CLIPEmbedder:
    """Generates CLIP embeddings for images."""
    
    def __init__(self, model_path: str, config: ImageProcessingConfig):
        self.config = config
        
        if model_path and os.path.exists(model_path):
            print(f"✅ Loading CLIP from {model_path}")
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
            self.available = True
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"   Device: {self.device}")
        else:
            print("⚠️  CLIP not available")
            self.model = None
            self.processor = None
            self.available = False
    
    def embed(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for image."""
        if not self.available:
            return np.random.randn(self.config.embedding_dim).astype('float32')
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            
            # Normalize
            embedding_np = embedding.cpu().numpy().astype('float32')
            embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)
            
            return embedding_np.flatten()
        
        except Exception as e:
            print(f"⚠️  CLIP embedding failed: {e}")
            return np.random.randn(self.config.embedding_dim).astype('float32')


class MetadataExtractor:
    """Extracts metadata from image files."""
    
    @staticmethod
    def extract(image_path: str) -> Dict[str, any]:
        """Extract comprehensive metadata."""
        try:
            image = Image.open(image_path)
            stat = os.stat(image_path)
            
            # EXIF data
            exif = image.getexif() or {}
            exif_data = {}
            for k, v in exif.items():
                try:
                    exif_data[str(k)] = str(v)
                except:
                    pass
            
            return {
                'filename': os.path.basename(image_path),
                'format': image.format,
                'mode': image.mode,
                'width': image.size[0],
                'height': image.size[1],
                'file_size_bytes': stat.st_size,
                'file_size_mb': float(stat.st_size / (1024 * 1024)),
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'has_exif': len(exif_data) > 0,
                'exif_data': exif_data
            }
        
        except Exception as e:
            return {
                'filename': os.path.basename(image_path),
                'error': str(e)
            }


class ContentAnalyzer:
    """Analyzes image content to classify type."""
    
    @staticmethod
    def analyze(ocr_result: Dict, screenshot_info: Dict, metadata: Dict) -> Dict[str, any]:
        """
        Analyze content and classify image type.
        
        Returns:
            {
                'primary_type': str,
                'sub_type': str,
                'content_description': str,
                'keywords': List[str]
            }
        """
        text = ocr_result['text'].lower()
        word_count = ocr_result['word_count']
        is_screenshot = screenshot_info['is_screenshot']
        
        # Classify primary type
        if is_screenshot:
            # Determine screenshot sub-type
            if 'email' in text or 'inbox' in text or '@' in text:
                primary_type = "screenshot"
                sub_type = "email"
            elif 'chat' in text or 'message' in text:
                primary_type = "screenshot"
                sub_type = "chat"
            elif 'chrome' in text or 'browser' in text or 'http' in text:
                primary_type = "screenshot"
                sub_type = "browser"
            else:
                primary_type = "screenshot"
                sub_type = "application"
        
        elif word_count > 50:
            # Text-heavy
            primary_type = "document"
            if word_count > 200:
                sub_type = "scan"
            else:
                sub_type = "snippet"
        
        elif word_count > 10:
            # Some text, likely diagram
            if any(word in text for word in ['chart', 'graph', 'figure', 'diagram']):
                primary_type = "diagram"
                sub_type = "labeled"
            else:
                primary_type = "mixed"
                sub_type = "text_visual"
        
        else:
            # Minimal text, visual content
            primary_type = "photo"
            sub_type = "visual_only"
        
        # Extract keywords
        words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ chars
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their'}
        keywords = [w for w in words if w not in stop_words]
        keyword_freq = Counter(keywords)
        top_keywords = [word for word, count in keyword_freq.most_common(10)]
        
        # Generate description
        if primary_type == "screenshot":
            description = f"{sub_type.title()} screenshot"
            if screenshot_info['indicators'].get('applications'):
                description += f" from {screenshot_info['indicators']['applications'][0]}"
        elif primary_type == "document":
            description = f"Document {sub_type} with {word_count} words"
        elif primary_type == "diagram":
            description = f"Diagram or chart with labels"
        elif primary_type == "photo":
            description = f"Photo or visual image"
        else:
            description = f"{primary_type.title()} content"
        
        return {
            'primary_type': primary_type,
            'sub_type': sub_type,
            'content_description': description,
            'keywords': top_keywords,
            'has_rich_text': word_count > 50,
            'has_minimal_text': 0 < word_count <= 10,
            'text_only': word_count > 200 and not is_screenshot
        }


class ImageProcessor:
    """Main image processing orchestrator."""
    
    def __init__(
        self,
        clip_model_path: str = None,
        config: ImageProcessingConfig = None
    ):
        self.config = config or ImageProcessingConfig()
        
        # Initialize components
        self.ocr_engine = OCREngine(self.config)
        self.screenshot_detector = ScreenshotDetector(self.config)
        self.clip_embedder = CLIPEmbedder(clip_model_path, self.config)
        self.metadata_extractor = MetadataExtractor()
        self.content_analyzer = ContentAnalyzer()
    
    def process_image(
        self,
        image_path: str,
        user_description: str = "",
        timestamp: str = None,
        session_id: str = None,
        source_document: str = None,
        figure_number: str = None,
        figure_title: str = None
    ) -> Dict[str, any]:
        """
        Process a single image completely.
        
        Returns:
            Complete chunk ready for vectorstore
        """
        print(f"\n🖼️  Processing: {os.path.basename(image_path)}")
        
        # Step 1: Extract metadata
        metadata = self.metadata_extractor.extract(image_path)
        print(f"   ✓ Metadata: {metadata.get('width', 0)}x{metadata.get('height', 0)} {metadata.get('format', '?')}")
        
        # Step 2: OCR
        ocr_result = self.ocr_engine.extract_text(image_path)
        print(f"   ✓ OCR: {ocr_result['word_count']} words ({ocr_result['confidence']:.1f}%)")
        
        # Step 3: Screenshot detection
        screenshot_info = self.screenshot_detector.detect(image_path, ocr_result)
        if screenshot_info['is_screenshot']:
            print(f"   ✓ Screenshot: {screenshot_info['confidence']:.0%} confidence")
            if screenshot_info['indicators'].get('applications'):
                print(f"     - Apps: {', '.join(screenshot_info['indicators']['applications'])}")
        
        # Step 4: Content analysis
        content_analysis = self.content_analyzer.analyze(ocr_result, screenshot_info, metadata)
        print(f"   ✓ Type: {content_analysis['primary_type']}/{content_analysis['sub_type']}")
        
        # Step 5: CLIP embedding
        embedding = self.clip_embedder.embed(image_path)
        print(f"   ✓ Embedding: {embedding.shape}")
        
        # Step 6: Build content string
        content = self._build_content(
            image_path, metadata, ocr_result, screenshot_info,
            content_analysis, user_description, timestamp,
            figure_number, figure_title
        )
        
        # Step 7: Build metadata
        chunk_metadata = self._build_metadata(
            image_path, metadata, ocr_result, screenshot_info,
            content_analysis, user_description, timestamp,
            session_id, source_document, figure_number, figure_title,
            embedding
        )
        
        # Step 8: Return chunk
        return {
            'id': f"image_{Path(image_path).stem}_{int(datetime.now().timestamp())}",
            'content': content,
            'embedding': embedding,
            'metadata': chunk_metadata,
            'token_count': len(content.split())
        }
    
    def _build_content(
        self, image_path, metadata, ocr_result, screenshot_info,
        content_analysis, user_description, timestamp,
        figure_number, figure_title
    ) -> str:
        """Build content string for chunk."""
        parts = []
        
        # Extract meaningful subject from filename
        raw_stem = Path(image_path).stem
        clean_stem = re.sub(r'[-_][0-9a-fA-F]{6,}$', '', raw_stem)
        clean_name = re.sub(r'[-_]+', ' ', clean_stem).strip().title()
        is_generic = clean_name.lower() in ('image', 'img', 'photo', 'picture', 'file', 'upload', 'screenshot', 'sample')

        # Title & Subject
        if figure_title:
            parts.append(f"# {figure_title}\n")
        elif figure_number:
            parts.append(f"# Image {figure_number}: {clean_name}\n")
        elif not is_generic and len(clean_name) >= 2:
            parts.append(f"# Image: {clean_name} ({metadata['filename']})\n")
            parts.append(f"**Subject:** {clean_name}\n")
        else:
            parts.append(f"# {content_analysis['content_description']}\n")
        
        # User description or auto visual description
        if user_description:
            parts.append(f"**Description:** {user_description}\n")
        elif not is_generic and len(clean_name) >= 2:
            parts.append(f"**Visual Content:** This is an image of {clean_name}.\n")
        
        # File info
        parts.append(f"**File:** {metadata['filename']}")
        parts.append(f"**Size:** {metadata['width']}x{metadata['height']} pixels")
        parts.append(f"**Type:** {content_analysis['primary_type']}/{content_analysis['sub_type']}")
        
        # Timestamp
        if timestamp:
            parts.append(f"**Timestamp:** {timestamp}")
        elif screenshot_info['indicators'].get('timestamps'):
            parts.append(f"**Detected Time:** {screenshot_info['indicators']['timestamps'][0]}")
        
        # Screenshot specific
        if screenshot_info['is_screenshot']:
            if screenshot_info['indicators'].get('applications'):
                parts.append(f"**Application:** {', '.join(screenshot_info['indicators']['applications'])}")
        
        # OCR text
        if ocr_result['has_text']:
            parts.append(f"\n**Extracted Text ({ocr_result['confidence']:.0f}% confidence):**")
            parts.append(ocr_result['text'])
        elif ocr_result['word_count'] > 0:
            parts.append(f"\n**Partial Text ({ocr_result['confidence']:.0f}% confidence):**")
            parts.append(ocr_result['text'])
        
        # Keywords
        if content_analysis['keywords']:
            parts.append(f"\n**Keywords:** {', '.join(content_analysis['keywords'][:10])}")
        
        return "\n".join(parts)
    
    def _build_metadata(
        self, image_path, metadata, ocr_result, screenshot_info,
        content_analysis, user_description, timestamp, session_id,
        source_document, figure_number, figure_title, embedding
    ) -> Dict:
        """Build metadata dictionary."""
        raw_stem = Path(image_path).stem
        clean_stem = re.sub(r'[-_][0-9a-fA-F]{6,}$', '', raw_stem)
        clean_name = re.sub(r'[-_]+', ' ', clean_stem).strip().title()
        is_generic = clean_name.lower() in ('image', 'img', 'photo', 'picture', 'file', 'upload', 'screenshot', 'sample')

        return {
            # Type info
            'type': 'image',
            'modality': 'image',
            'primary_type': content_analysis['primary_type'],
            'sub_type': content_analysis['sub_type'],
            'subject': clean_name if not is_generic else '',
            
            # Source info
            'source': source_document or metadata['filename'],
            'image_path': image_path,
            'filename': metadata['filename'],
            'session_id': session_id,
            
            # Figure info
            'figure_number': figure_number,
            'figure_title': figure_title,
            
            # Image properties
            'image_width': int(metadata['width']),
            'image_height': int(metadata['height']),
            'image_format': metadata['format'],
            'file_size_mb': float(metadata['file_size_mb']),
            
            # OCR results
            'has_ocr_text': bool(ocr_result['has_text']),
            'ocr_confidence': float(ocr_result['confidence']),
            'ocr_word_count': int(ocr_result['word_count']),
            'ocr_method': ocr_result['method'],
            
            # Screenshot info
            'is_screenshot': bool(screenshot_info['is_screenshot']),
            'screenshot_confidence': float(screenshot_info['confidence']),
            'detected_timestamps': screenshot_info['indicators'].get('timestamps', []),
            'detected_applications': screenshot_info['indicators'].get('applications', []),
            
            # Content analysis
            'content_description': content_analysis['content_description'],
            'keywords': content_analysis['keywords'],
            'has_rich_text': bool(content_analysis['has_rich_text']),
            
            # Timestamps
            'timestamp': timestamp,
            'created_time': metadata['created_time'],
            'modified_time': metadata['modified_time'],
            
            # User input
            'user_description': user_description,
            
            # Embedding info
            'embedding_type': 'clip',
            'embedding_dim': int(len(embedding)),
            'use_image_directly': True,
            'has_image_file': True,
            
            # Searchable references
            'reference_names': list(filter(None, [
                figure_title,
                clean_name if not is_generic else None,
                f"image {clean_name}" if not is_generic else None,
                metadata['filename'],
                content_analysis['content_description']
            ])),
            
            # Full searchable text
            'searchable_text': ' '.join(filter(None, [
                clean_name if not is_generic else '',
                f"image of {clean_name}" if not is_generic else '',
                metadata['filename'],
                figure_title or '',
                user_description,
                ocr_result['text'],
                content_analysis['content_description'],
                ' '.join(content_analysis['keywords'])
            ]))
        }

    
    def process_batch(
        self,
        image_paths: List[str],
        session_id: str = None,
        output_file: str = "processed_images.pkl"
    ) -> List[Dict]:
        """Process multiple images."""
        all_chunks = []
        
        print(f"\n{'='*70}")
        print(f"📸 PROCESSING {len(image_paths)} IMAGES")
        print(f"{'='*70}")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}]", end=" ")
            try:
                chunk = self.process_image(
                    image_path=image_path,
                    figure_number=str(i),
                    session_id=session_id
                )
                all_chunks.append(chunk)
                print(f"   ✅ Success")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # Save
        self._save_results(all_chunks, output_file, session_id)
        
        return all_chunks
    
    def _save_results(self, chunks: List[Dict], output_file: str, session_id: str):
        """Save processing results."""
        print(f"\n{'='*70}")
        print(f"💾 SAVING RESULTS")
        print(f"{'='*70}")
        
        # Save pickle (full data)
        with open(output_file, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"✅ Saved chunks: {output_file}")
        
        # Save JSON summary
        summary = {
            'total_images': len(chunks),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'statistics': self._calculate_stats(chunks),
            'images': [
                {
                    'id': c['id'],
                    'filename': c['metadata']['filename'],
                    'type': f"{c['metadata']['primary_type']}/{c['metadata']['sub_type']}",
                    'is_screenshot': c['metadata']['is_screenshot'],
                    'has_text': c['metadata']['has_ocr_text'],
                    'ocr_confidence': c['metadata']['ocr_confidence'],
                    'word_count': c['metadata']['ocr_word_count']
                }
                for c in chunks
            ]
        }
        
        summary_file = output_file.replace('.pkl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved summary: {summary_file}")
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING STATISTICS")
        print(f"{'='*70}")
        stats = summary['statistics']
        print(f"   Total images: {stats['total']}")
        print(f"   Screenshots: {stats['screenshots']}")
        print(f"   With OCR text: {stats['with_text']}")
        print(f"   Average confidence: {stats['avg_confidence']:.1f}%")
        print(f"\n   Types:")
        for type_name, count in stats['types'].items():
            print(f"     - {type_name}: {count}")
        print(f"{'='*70}\n")
    
    def _calculate_stats(self, chunks: List[Dict]) -> Dict:
        """Calculate statistics from processed chunks."""
        types = Counter(
            f"{c['metadata']['primary_type']}/{c['metadata']['sub_type']}"
            for c in chunks
        )
        
        confidences = [c['metadata']['ocr_confidence'] for c in chunks]
        
        return {
            'total': len(chunks),
            'screenshots': sum(1 for c in chunks if c['metadata']['is_screenshot']),
            'with_text': sum(1 for c in chunks if c['metadata']['has_ocr_text']),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0,
            'types': dict(types)
        }


def process_directory(
    directory_path: str,
    clip_model_path: str = None,
    session_id: str = None,
    output_file: str = None,
    recursive: bool = True
) -> List[Dict]:
    """
    Process all images in a directory.
    
    Args:
        directory_path: Path to directory containing images
        clip_model_path: Path to CLIP model
        session_id: Optional session identifier
        output_file: Output file path
        recursive: Process subdirectories recursively
        
    Returns:
        List of processed chunks
    """
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    image_paths = []
    
    if recursive:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory_path):
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(directory_path, file))
    
    if not image_paths:
        print(f"❌ No images found in {directory_path}")
        return []
    
    print(f"✅ Found {len(image_paths)} images in {directory_path}")
    
    # Sort by filename
    image_paths.sort()
    
    # Initialize processor
    processor = ImageProcessor(
        clip_model_path=clip_model_path,
        config=ImageProcessingConfig()
    )
    
    # Set output file
    if output_file is None:
        dir_name = Path(directory_path).name
        output_file = f"images_{dir_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    # Set session ID
    if session_id is None:
        session_id = Path(directory_path).name
    
    # Process images
    chunks = processor.process_batch(
        image_paths=image_paths,
        session_id=session_id,
        output_file=output_file
    )
    
    return chunks


def process_with_context(
    image_path: str,
    context_before: str = "",
    context_after: str = "",
    clip_model_path: str = None,
    **kwargs
) -> Dict:
    """
    Process image with surrounding context (for PDF/DOCX images).
    
    Args:
        image_path: Path to image
        context_before: Text before image
        context_after: Text after image
        clip_model_path: Path to CLIP model
        **kwargs: Additional metadata
        
    Returns:
        Processed chunk
    """
    processor = ImageProcessor(clip_model_path=clip_model_path)
    
    # Process image
    chunk = processor.process_image(image_path, **kwargs)
    
    # Add context to content
    content_parts = [chunk['content']]
    
    if context_before:
        content_parts.insert(0, f"**Context Before:**\n{context_before}\n")
    
    if context_after:
        content_parts.append(f"\n**Context After:**\n{context_after}")
    
    chunk['content'] = "\n".join(content_parts)
    chunk['metadata']['has_context'] = bool(context_before or context_after)
    chunk['metadata']['context_before'] = context_before
    chunk['metadata']['context_after'] = context_after
    
    return chunk


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images for multimodal RAG')
    parser.add_argument('path', help='Path to image or directory')
    parser.add_argument('--clip-model', default='/Users/vijay/Documents/SIH/clip-model',
                       help='Path to CLIP model')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--session', help='Session ID')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Do not process subdirectories')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        # Process directory
        chunks = process_directory(
            directory_path=args.path,
            clip_model_path=args.clip_model,
            session_id=args.session,
            output_file=args.output,
            recursive=not args.no_recursive
        )
    else:
        # Process single image
        processor = ImageProcessor(clip_model_path=args.clip_model)
        chunk = processor.process_image(args.path)
        
        output_file = args.output or "single_image.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump([chunk], f)
        
        print(f"\n✅ Processed 1 image")
        print(f"   Type: {chunk['metadata']['primary_type']}/{chunk['metadata']['sub_type']}")
        print(f"   OCR: {chunk['metadata']['ocr_word_count']} words ({chunk['metadata']['ocr_confidence']:.1f}%)")
        print(f"   Saved to: {output_file}")
        
        chunks = [chunk]
    
    print(f"\n✅ All done! Processed {len(chunks)} images.")
    print(f"📦 Ready for vectorstore.py!")


if __name__ == "__main__":
    main()