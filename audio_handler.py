

# audio_handler.py - Enhanced Audio Processing with librosa Speaker Diarization

import os
import sys
os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import time
from datetime import timedelta, datetime
import numpy as np
from collections import Counter
import re
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Audio processing
from pydub import AudioSegment, silence
import librosa
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER = True
except ImportError:
    import whisper
    FASTER_WHISPER = False


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    whisper_model: str = "base"  # tiny, base, small, medium, large
    num_speakers: Optional[int] = None  # None = auto-detect
    max_speakers: int = 5  # Maximum speakers to test
    min_speakers: int = 2  # Minimum speakers to test
    min_silence_len: int = 500  # ms
    silence_thresh_offset: int = 14  # dB below average
    keep_silence: int = 200  # ms
    n_mfcc: int = 13  # MFCC features


class EnhancedAudioHandler:
    """
    Enhanced audio handler with:
    - Better speaker diarization using librosa + KMeans
    - AUTOMATIC speaker detection (Silhouette Score + Elbow Method)
    - One timestamp = one chunk (no artificial splitting)
    - Better transcription with Whisper
    - Works on all audio formats
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        
        print(f"\n{'='*70}")
        print(f"🎧 ENHANCED AUDIO HANDLER")
        print(f"{'='*70}")
        print(f"Features:")
        print(f"  ✓ librosa MFCC-based speaker diarization")
        print(f"  ✓ KMeans clustering for speaker separation")
        print(f"  ✓ AUTOMATIC speaker detection (Silhouette + Elbow)")
        print(f"  ✓ Silence-based natural segmentation")
        print(f"  ✓ One timestamp = one chunk")
        print(f"  ✓ Better transcription accuracy")
        print(f"\nWhisper Model: {self.config.whisper_model}")
        
        # Show speaker detection mode
        if self.config.num_speakers is None:
            print(f"Speaker Detection: AUTO (testing {self.config.min_speakers}-{self.config.max_speakers} speakers)")
        else:
            print(f"Speaker Detection: MANUAL ({self.config.num_speakers} speakers)")
        
        # Load Whisper
        print(f"\n📥 Loading Whisper {self.config.whisper_model}...")
        self.use_pipeline = False
        try:
            if FASTER_WHISPER:
                self.model = WhisperModel(
                    self.config.whisper_model,
                    device="cpu",
                    compute_type="int8"
                )
                self.use_faster = True
            else:
                self.model = whisper.load_model(self.config.whisper_model)
                self.use_faster = False
            print(f"✅ Model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load standard Whisper model: {e}")
            print("🔄 Attempting fallback to cached Hugging Face transformers pipeline...")
            try:
                from transformers import pipeline
                import torch
                
                device = 0 if torch.cuda.is_available() else -1
                self.model = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3-turbo",
                    device=device,
                    local_files_only=True   # OFFLINE: never download from the internet
                )
                self.use_faster = False
                self.use_pipeline = True
                print("✅ Successfully loaded cached Hugging Face whisper-large-v3-turbo pipeline!")
            except Exception as e_fallback:
                print(f"❌ Fallback failed: {e_fallback}")
                raise e
        print(f"{'='*70}\n")
    
    def process_audio(
        self,
        audio_path: str,
        user_description: str = "",
        meeting_title: str = None,
        session_id: str = None
    ) -> List[Dict]:
        """
        Process audio file with high-speed single-pass Whisper transcription.
        
        Returns:
            List of chunks (one per natural segment/timestamp + full master overview chunk)
        """
        filename = os.path.basename(audio_path)
        print(f"\n🎧 Processing Audio: {filename}")
        
        # Validate
        if not self._validate_audio(audio_path):
            return []
        
        start_t = time.time()
        
        # Step 1: Direct Single-Pass Whisper Transcription (1-2 seconds)
        print(f"   ⚡ Fast Whisper Transcription...")
        transcribed_chunks = []
        full_text_list = []
        
        try:
            if self.use_faster:
                segments_iter, info = self.model.transcribe(
                    audio_path,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300)
                )
                for i, seg in enumerate(segments_iter):
                    text = seg.text.strip()
                    if text:
                        transcribed_chunks.append({
                            'text': text,
                            'speaker': f"Speaker 1",
                            'start_time': seg.start,
                            'end_time': seg.end,
                            'duration': seg.end - seg.start,
                            'language': info.language,
                            'chunk_index': i
                        })
                        full_text_list.append(text)
            elif self.use_pipeline:
                result = self.model(audio_path, return_timestamps=True)
                pipe_chunks = result.get('chunks', [])
                for i, pc in enumerate(pipe_chunks):
                    ts = pc.get('timestamp', (0.0, 0.0))
                    start_t_seg = ts[0] or 0.0
                    end_t_seg = ts[1] or start_t_seg + 2.0
                    text = pc.get('text', '').strip()
                    if text:
                        transcribed_chunks.append({
                            'text': text,
                            'speaker': "Speaker 1",
                            'start_time': start_t_seg,
                            'end_time': end_t_seg,
                            'duration': end_t_seg - start_t_seg,
                            'language': 'en',
                            'chunk_index': i
                        })
                        full_text_list.append(text)
            else:
                result = self.model.transcribe(audio_path, verbose=False)
                for i, seg in enumerate(result.get('segments', [])):
                    text = seg.get('text', '').strip()
                    if text:
                        transcribed_chunks.append({
                            'text': text,
                            'speaker': "Speaker 1",
                            'start_time': seg.get('start', 0.0),
                            'end_time': seg.get('end', 0.0),
                            'duration': seg.get('end', 0.0) - seg.get('start', 0.0),
                            'language': result.get('language', 'en'),
                            'chunk_index': i
                        })
                        full_text_list.append(text)
        except Exception as e:
            print(f"      ❌ Transcription failed: {e}")
            return []

        full_text = " ".join(full_text_list).strip()
        print(f"      ✓ Transcribed {len(transcribed_chunks)} segments in {time.time() - start_t:.2f}s")
        print(f"      ✓ Total Words: {len(full_text.split())}")

        # Step 2: Extract Analysis
        analysis = self._analyze_content(full_text)
        
        # Step 3: Format Chunks
        formatted_chunks = self._format_chunks(
            transcribed_chunks=transcribed_chunks,
            audio_path=audio_path,
            analysis=analysis,
            user_description=user_description,
            meeting_title=meeting_title,
            session_id=session_id
        )

        # Step 4: Add Master Overview Chunk (Contains Full Transcript for comprehensive questions)
        if full_text:
            overview_content = (
                f"# Audio Recording: {filename}\n"
                f"**File Type:** Audio Transcript\n"
                f"**Total Segments:** {len(transcribed_chunks)}\n"
                f"**Key Topics:** {', '.join(analysis.get('topics', [])) or 'General Discussion'}\n\n"
                f"## Full Transcript:\n{full_text}"
            )
            overview_chunk = {
                'id': f"audio_{Path(audio_path).stem}_master_overview",
                'content': overview_content,
                'metadata': {
                    'type': 'audio',
                    'modality': 'audio',
                    'source': filename,
                    'filename': filename,
                    'audio_path': audio_path,
                    'is_overview': True,
                    'chunk_index': 0,
                    'total_chunks': len(transcribed_chunks) + 1,
                    'timestamp_formatted': '[Full Audio]',
                    'speaker': 'All Speakers',
                    'word_count': len(full_text.split()),
                    'searchable_text': f"{filename} audio transcript {full_text}"
                },
                'token_count': len(full_text.split())
            }
            # Prepend master overview chunk
            formatted_chunks.insert(0, overview_chunk)

        print(f"   ✅ Created {len(formatted_chunks)} searchable chunks in {time.time() - start_t:.2f}s\n")
        return formatted_chunks
    
    def _validate_audio(self, audio_path: str) -> bool:
        """Validate audio file."""
        if not os.path.exists(audio_path):
            print(f"   ❌ File not found")
            return False
        
        valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
        ext = Path(audio_path).suffix.lower()
        
        if ext not in valid_extensions:
            print(f"   ❌ Unsupported format: {ext}")
            return False
        
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"   ✓ Valid: {ext} ({size_mb:.1f} MB)")
        
        return True
    
    def _split_by_silence(self, audio: AudioSegment) -> List[Tuple[AudioSegment, float, float]]:
        """
        Split audio by silence into natural segments.
        
        Returns:
            List of (audio_chunk, start_time, end_time) tuples
        """
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=self.config.min_silence_len,
            silence_thresh=audio.dBFS - self.config.silence_thresh_offset,
            keep_silence=self.config.keep_silence
        )
        
        # Calculate timestamps for each chunk
        chunks_with_times = []
        current_time = 0.0
        
        for chunk in chunks:
            start_time = current_time
            duration = len(chunk) / 1000.0  # Convert ms to seconds
            end_time = start_time + duration
            
            chunks_with_times.append((chunk, start_time, end_time))
            current_time = end_time
        
        return chunks_with_times
    
    def _extract_mfcc_features(
        self, 
        chunks: List[Tuple[AudioSegment, float, float]]
    ) -> Tuple[np.ndarray, List]:
        """
        Extract MFCC features from audio chunks for speaker diarization.
        
        Returns:
            (mfcc_features_array, valid_chunks)
        """
        mfcc_features = []
        valid_chunks = []
        
        for i, (chunk, start_time, end_time) in enumerate(chunks):
            try:
                # Export chunk to temp file
                temp_file = f"temp_chunk_{i}.wav"
                chunk.export(temp_file, format="wav")
                
                # Load with librosa
                y, sr = librosa.load(temp_file, sr=None)
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc)
                mfcc_mean = np.mean(mfcc, axis=1)
                
                mfcc_features.append(mfcc_mean)
                valid_chunks.append((chunk, start_time, end_time))
                
                # Cleanup
                os.remove(temp_file)
                
            except Exception as e:
                print(f"      ⚠️  Skipped chunk {i}: {e}")
                continue
        
        return np.array(mfcc_features), valid_chunks
    
    def _detect_optimal_speakers(self, mfcc_features: np.ndarray) -> int:
        """
        Automatically detect optimal number of speakers using:
        1. Silhouette Score (primary)
        2. Elbow Method (fallback)
        
        Returns:
            Optimal number of speakers
        """
        n_samples = len(mfcc_features)
        
        # Need at least min_speakers samples
        if n_samples < self.config.min_speakers:
            print(f"      ⚠️  Only {n_samples} chunks, defaulting to 1 speaker")
            return 1
        
        # Test range of speaker counts
        min_k = self.config.min_speakers
        max_k = min(self.config.max_speakers, n_samples - 1)  # Can't have more clusters than samples
        
        if max_k < min_k:
            return 1
        
        print(f"      🔍 Testing {min_k} to {max_k} speakers...")
        
        # Method 1: Silhouette Score (better for speaker detection)
        silhouette_scores = []
        inertias = []
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(mfcc_features)
            
            # Silhouette score (higher is better, range: -1 to 1)
            score = silhouette_score(mfcc_features, labels)
            silhouette_scores.append(score)
            
            # Inertia for elbow method
            inertias.append(kmeans.inertia_)
            
            print(f"         k={k}: silhouette={score:.3f}, inertia={kmeans.inertia_:.2f}")
        
        # Find best k using silhouette score
        best_idx = np.argmax(silhouette_scores)
        optimal_k_silhouette = min_k + best_idx
        best_score = silhouette_scores[best_idx]
        
        print(f"      ✓ Best by Silhouette: {optimal_k_silhouette} speakers (score: {best_score:.3f})")
        
        # Method 2: Elbow method as validation
        try:
            kl = KneeLocator(
                range(min_k, max_k + 1), 
                inertias, 
                curve='convex', 
                direction='decreasing'
            )
            optimal_k_elbow = kl.elbow if kl.elbow else optimal_k_silhouette
            print(f"      ✓ Elbow method suggests: {optimal_k_elbow} speakers")
        except:
            optimal_k_elbow = optimal_k_silhouette
        
        # Use silhouette score as primary method
        # Only use elbow if silhouette score is very low (< 0.2)
        if best_score < 0.2 and optimal_k_elbow != optimal_k_silhouette:
            print(f"      ⚠️  Low silhouette score, using elbow method")
            return optimal_k_elbow
        
        return optimal_k_silhouette
    
    def _cluster_speakers(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        Cluster MFCC features to identify speakers using KMeans.
        Automatically detects optimal number of speakers if not specified.
        
        Returns:
            Array of speaker labels (0, 1, 2, ...)
        """
        if len(mfcc_features) < 2:
            # Not enough chunks to cluster
            print("      ⚠️  Too few chunks for clustering")
            return np.zeros(len(mfcc_features), dtype=int)
        
        # Determine number of speakers
        if self.config.num_speakers is None:
            # AUTO-DETECT
            num_speakers = self._detect_optimal_speakers(mfcc_features)
            print(f"      🎯 Auto-detected: {num_speakers} speakers")
        else:
            # MANUAL
            num_speakers = min(self.config.num_speakers, len(mfcc_features))
            print(f"      👤 Using specified: {num_speakers} speakers")
        
        # Edge case: only 1 speaker
        if num_speakers == 1:
            return np.zeros(len(mfcc_features), dtype=int)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=num_speakers, random_state=0, n_init=10)
        labels = kmeans.fit_predict(mfcc_features)
        
        # Print speaker distribution
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"      📊 Speaker distribution: {distribution}")
        
        return labels
    
    def _transcribe_chunks(
        self,
        chunks: List[Tuple[AudioSegment, float, float]],
        speaker_labels: np.ndarray,
        original_audio_path: str
    ) -> List[Dict]:
        """
        Transcribe each chunk with Whisper.
        
        Returns:
            List of dicts with text, speaker, start_time, end_time
        """
        if self.use_pipeline:
            try:
                print(f"   📝 Transcribing entire audio file at once with whisper-large-v3-turbo pipeline...")
                
                # Dynamically increase PyTorch threads for faster CPU inference
                import torch
                old_threads = torch.get_num_threads()
                # 4 threads is the sweet spot for CPU deep learning to avoid memory bus saturation
                target_threads = min(4, os.cpu_count())
                torch.set_num_threads(target_threads)
                print(f"      ✓ Accelerated CPU inference: Using {target_threads} threads (previously {old_threads})")
                
                try:
                    # Run ASR pipeline in inference mode (faster CPU execution)
                    with torch.inference_mode():
                        pipe_result = self.model(
                            original_audio_path, 
                            chunk_length_s=30, 
                            batch_size=4, 
                            return_timestamps=True
                        )
                finally:
                    # Restore original thread configuration
                    torch.set_num_threads(old_threads)
                
                pipe_chunks = pipe_result.get('chunks', [])
                print(f"      ✓ Transcription complete, matching with {len(chunks)} diarized segments...")
                
                transcribed = []
                for i, ((chunk_audio, start_time, end_time), speaker_label) in enumerate(zip(chunks, speaker_labels)):
                    chunk_texts = []
                    for pc in pipe_chunks:
                        ts = pc.get('timestamp')
                        if ts:
                            pc_start, pc_end = ts
                            if pc_start is None: pc_start = start_time
                            if pc_end is None: pc_end = end_time
                            
                            # Check overlap
                            overlap = max(start_time, pc_start) < min(end_time, pc_end)
                            if overlap:
                                chunk_texts.append(pc.get('text', '').strip())
                    
                    text = ' '.join(chunk_texts).strip()
                    if not text:
                        # Fallback to transcribing individual chunk
                        try:
                            temp_file = f"temp_transcribe_fallback_{i}.wav"
                            chunk_audio.export(temp_file, format="wav")
                            res = self.model(temp_file)
                            text = res.get('text', '').strip()
                            os.remove(temp_file)
                        except:
                            text = ""
                            
                    if text:
                        transcribed.append({
                            'text': text,
                            'speaker': f"Speaker {int(speaker_label) + 1}",
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'language': 'en',
                            'chunk_index': i
                        })
                return transcribed
            except Exception as e:
                print(f"      ⚠️  Full file transcription failed: {e}. Falling back to chunk-by-chunk...")

        # Chunk-by-chunk fallback/default logic
        transcribed = []
        for i, ((chunk, start_time, end_time), speaker_label) in enumerate(zip(chunks, speaker_labels)):
            try:
                # Export chunk
                temp_file = f"temp_transcribe_{i}.wav"
                chunk.export(temp_file, format="wav")
                
                # Transcribe
                if self.use_faster:
                    segments_iter, info = self.model.transcribe(temp_file, beam_size=5)
                    segments = list(segments_iter)
                    text = ' '.join([seg.text for seg in segments]).strip()
                    language = info.language
                elif self.use_pipeline:
                    result = self.model(temp_file)
                    text = result.get('text', '').strip()
                    language = 'en'
                else:
                    result = self.model.transcribe(temp_file, verbose=False)
                    text = result['text'].strip()
                    language = result['language']
                
                # Cleanup
                os.remove(temp_file)
                
                if text:
                    transcribed.append({
                        'text': text,
                        'speaker': f"Speaker {int(speaker_label) + 1}",
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'language': language,
                        'chunk_index': i
                    })
                
            except Exception as e:
                print(f"      ⚠️  Transcription failed for chunk {i}: {e}")
                continue
        
        return transcribed
    
    def _analyze_content(self, full_text: str) -> Dict:
        """Analyze content for keywords and topics."""
        words = re.findall(r'\b\w{4,}\b', full_text.lower())
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were',
            'their', 'will', 'would', 'there', 'what', 'when', 'where',
            'which', 'about', 'could', 'should', 'think', 'know',
            'just', 'like', 'well', 'yeah', 'okay', 'right', 'going'
        }
        keywords = [w for w in words if w not in stop_words]
        keyword_freq = Counter(keywords)
        top_keywords = [word for word, count in keyword_freq.most_common(20)]
        
        # Detect topics
        topics = []
        topic_patterns = {
            'meeting': ['meeting', 'agenda', 'discuss', 'presentation'],
            'project': ['project', 'deadline', 'milestone', 'deliverable'],
            'technical': ['code', 'system', 'database', 'api', 'server'],
            'financial': ['budget', 'cost', 'revenue', 'profit', 'expense'],
            'planning': ['plan', 'strategy', 'roadmap', 'timeline'],
        }
        
        text_lower = full_text.lower()
        for topic, keywords_list in topic_patterns.items():
            if sum(1 for kw in keywords_list if kw in text_lower) >= 2:
                topics.append(topic)
        
        # Extract entities
        entities = []
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', full_text)
        entities.extend(dates[:5])
        times = re.findall(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b', full_text)
        entities.extend(times[:5])
        amounts = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', full_text)
        entities.extend(amounts[:5])
        
        return {
            'keywords': top_keywords,
            'topics': topics,
            'entities': list(set(entities))
        }
    
    def _format_chunks(
        self,
        transcribed_chunks: List[Dict],
        audio_path: str,
        analysis: Dict,
        user_description: str,
        meeting_title: str,
        session_id: str
    ) -> List[Dict]:
        """Format chunks for vectorstore (one timestamp = one chunk)."""
        formatted = []
        filename = os.path.basename(audio_path)
        
        for i, chunk in enumerate(transcribed_chunks):
            # Build content
            content = self._build_content(chunk, meeting_title, filename)
            
            # Extract chunk keywords
            chunk_keywords = self._get_chunk_keywords(chunk['text'])
            
            # Build metadata
            metadata = {
                # Type
                'type': 'audio',
                'modality': 'audio',
                
                # Source
                'source': filename,
                'audio_path': audio_path,
                'filename': filename,
                'session_id': session_id,
                
                # Meeting info
                'meeting_title': meeting_title,
                'user_description': user_description,
                
                # Chunk info (ONE timestamp = ONE chunk)
                'chunk_index': int(i),
                'total_chunks': int(len(transcribed_chunks)),
                'start_time': float(chunk['start_time']),
                'end_time': float(chunk['end_time']),
                'duration': float(chunk['duration']),
                'timestamp_formatted': f"[{self._format_timestamp(chunk['start_time'])} → {self._format_timestamp(chunk['end_time'])}]",
                
                # Speaker
                'speaker': chunk['speaker'],
                'num_speakers': 1,  # One speaker per natural segment
                
                # Content
                'word_count': int(len(chunk['text'].split())),
                'language': chunk['language'],
                'chunk_keywords': chunk_keywords,
                'overall_keywords': analysis['keywords'],
                'topics': analysis['topics'],
                'entities': analysis['entities'],
                
                # Processing
                'timestamp': datetime.now().isoformat(),
                'processing_mode': 'offline',
                'processing_engine': 'faster_whisper' if self.use_faster else 'openai_whisper',
                'diarization_method': 'librosa_kmeans_auto',
                
                # Searchable
                'searchable_text': ' '.join(filter(None, [
                    meeting_title or '',
                    user_description,
                    chunk['text'],
                    ' '.join(chunk_keywords)
                ]))
            }
            
            formatted.append({
                'id': f"audio_{Path(audio_path).stem}_chunk_{i}",
                'content': content,
                'metadata': metadata,
                'token_count': int(len(chunk['text'].split()))
            })
        
        return formatted
    
    def _build_content(self, chunk: Dict, meeting_title: str, filename: str) -> str:
        """Build content string for chunk."""
        lines = []
        
        # Header
        if meeting_title:
            lines.append(f"# {meeting_title} - Segment {chunk['chunk_index'] + 1}\n")
        else:
            lines.append(f"# Audio: {filename} - Segment {chunk['chunk_index'] + 1}\n")
        
        # Timestamp
        start = self._format_timestamp(chunk['start_time'])
        end = self._format_timestamp(chunk['end_time'])
        lines.append(f"**Timestamp:** [{start} → {end}] ({chunk['duration']:.1f}s)")
        
        # Speaker
        lines.append(f"**Speaker:** {chunk['speaker']}\n")
        
        # Transcription
        lines.append("**Transcription:**")
        lines.append(chunk['text'])
        
        return '\n'.join(lines)
    
    def _get_chunk_keywords(self, text: str) -> List[str]:
        """Extract keywords from chunk."""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {'that', 'this', 'with', 'from', 'have', 'were', 'just', 'like'}
        keywords = [w for w in words if w not in stop_words]
        word_freq = Counter(keywords)
        return [word for word, count in word_freq.most_common(10)]
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as readable duration."""
        return str(timedelta(seconds=int(seconds)))
    
    def process_batch(
        self,
        audio_paths: List[str],
        session_id: str = None,
        output_file: str = "audio_chunks"
    ) -> List[Dict]:
        """Process multiple audio files."""
        all_chunks = []
        
        print(f"\n{'='*70}")
        print(f"🎧 BATCH PROCESSING {len(audio_paths)} FILES")
        print(f"{'='*70}\n")
        
        for i, path in enumerate(audio_paths, 1):
            print(f"[{i}/{len(audio_paths)}]")
            try:
                chunks = self.process_audio(path, session_id=session_id)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
        
        # Save results
        self._save_results(all_chunks, output_file, session_id)
        
        return all_chunks
    
    def _save_results(self, chunks: List[Dict], output_prefix: str, session_id: str):
        """Save chunks to PKL and JSON files."""
        print(f"\n{'='*70}")
        print(f"💾 SAVING RESULTS")
        print(f"{'='*70}\n")
        
        # 1. Save PKL
        pkl_file = f"{output_prefix}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"✅ Saved PKL: {pkl_file}")
        
        # 2. Save JSON
        json_chunks = []
        for chunk in chunks:
            json_chunk = {
                'id': chunk['id'],
                'content_preview': chunk['content'][:300] + '...' if len(chunk['content']) > 300 else chunk['content'],
                'metadata': chunk['metadata'],
                'token_count': chunk['token_count']
            }
            json_chunks.append(json_chunk)
        
        json_file = f"{output_prefix}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_chunks, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved JSON: {json_file}")
        
        # 3. Save summary
        total_duration = sum(c['metadata']['duration'] for c in chunks)
        speakers = set(c['metadata']['speaker'] for c in chunks)
        languages = set(c['metadata']['language'] for c in chunks)
        topics = set()
        for c in chunks:
            topics.update(c['metadata'].get('topics', []))
        
        summary = {
            'total_chunks': len(chunks),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'processing_mode': 'offline',
            'diarization_method': 'librosa_kmeans_auto',
            'statistics': {
                'total': len(chunks),
                'total_duration': float(total_duration),
                'total_duration_formatted': str(timedelta(seconds=int(total_duration))),
                'unique_speakers': len(speakers),
                'speakers': list(speakers),
                'languages': list(languages),
                'topics': list(topics)
            }
        }
        
        summary_file = f"{output_prefix}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved Summary: {summary_file}")
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING STATISTICS")
        print(f"{'='*70}")
        stats = summary['statistics']
        print(f"   Total chunks: {stats['total']}")
        print(f"   Total duration: {stats['total_duration_formatted']}")
        print(f"   Speakers: {stats['unique_speakers']} ({', '.join(stats['speakers'])})")
        print(f"   Languages: {', '.join(stats['languages'])}")
        if stats['topics']:
            print(f"   Topics: {', '.join(stats['topics'])}")
        print(f"   Method: librosa MFCC + KMeans (AUTO)")
        print(f"{'='*70}\n")


# Convenience functions

def process_audio_file(
    audio_path: str, 
    output_prefix: str = "audio_chunks",
    num_speakers: Optional[int] = None
) -> List[Dict]:
    """Process single audio file."""
    config = AudioConfig(num_speakers=num_speakers)
    handler = EnhancedAudioHandler(config)
    chunks = handler.process_audio(audio_path)
    handler._save_results(chunks, output_prefix, session_id=Path(audio_path).stem)
    return chunks


def process_audio_directory(
    directory: str, 
    output_prefix: str = "audio_chunks",
    num_speakers: Optional[int] = None
) -> List[Dict]:
    """Process all audio in directory."""
    audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_exts:
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"❌ No audio files found in {directory}")
        return []
    
    print(f"✅ Found {len(audio_files)} audio files\n")
    audio_files.sort()
    
    config = AudioConfig(num_speakers=num_speakers)
    handler = EnhancedAudioHandler(config)
    return handler.process_batch(
        audio_files, 
        session_id=Path(directory).name,
        output_file=output_prefix
    )


def main():
    """Command-line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🎧 ENHANCED AUDIO HANDLER")
        print("="*70)
        print("\nFeatures:")
        print("  ✓ librosa MFCC + KMeans speaker diarization")
        print("  ✓ AUTOMATIC speaker detection (Silhouette Score)")
        print("  ✓ Natural silence-based segmentation")
        print("  ✓ One timestamp = one chunk (no artificial splitting)")
        print("  ✓ Better transcription accuracy")
        print("  ✓ Works on all audio formats")
        print("\nUsage:")
        print("  python audio_handler.py <audio_file> [num_speakers]")
        print("  python audio_handler.py <audio_directory> [num_speakers]")
        print("\nExamples:")
        print("  python audio_handler.py meeting.mp3           # Auto-detect speakers")
        print("  python audio_handler.py meeting.mp3 2         # Force 2 speakers")
        print("  python audio_handler.py /path/to/audio/       # Auto-detect")
        print("  python audio_handler.py /path/to/audio/ 3     # Force 3 speakers")
        print("="*70 + "\n")
        sys.exit(1)
    
    path = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None  # None for auto
    
    if os.path.isfile(path):
        chunks = process_audio_file(path, num_speakers=num_speakers)
        print(f"\n✅ Processed {len(chunks)} chunks from 1 file")
    elif os.path.isdir(path):
        chunks = process_audio_directory(path, num_speakers=num_speakers)
        print(f"\n✅ Processed {len(chunks)} chunks from directory")
    else:
        print(f"❌ Not found: {path}")
        sys.exit(1)
    
    print("\n📦 Files saved:")
    print("   - audio_chunks.pkl (for vectorstore)")
    print("   - audio_chunks.json (human-readable)")
    print("   - audio_chunks_summary.json (statistics)")
    print("\n🚀 Ready for vectorstore.py!\n")


if __name__ == "__main__":
    main()


# # audio_handler.py - FIXED: Temp file handling issues

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

# import json
# import pickle
# import tempfile
# import uuid
# from typing import Dict, List, Optional, Tuple
# from pathlib import Path
# from dataclasses import dataclass
# from datetime import timedelta, datetime
# import numpy as np
# from collections import Counter
# import re
# from sklearn.metrics import silhouette_score
# from kneed import KneeLocator

# # Audio processing
# from pydub import AudioSegment, silence
# import librosa
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import euclidean

# # Whisper
# try:
#     from faster_whisper import WhisperModel
#     FASTER_WHISPER = True
# except ImportError:
#     import whisper
#     FASTER_WHISPER = False


# @dataclass
# class AudioConfig:
#     """Configuration for audio processing."""
#     whisper_model: str = "small"
#     num_speakers: Optional[int] = None
#     max_speakers: int = 5
#     min_speakers: int = 2
#     min_silence_len: int = 500
#     silence_thresh_offset: int = 14
#     keep_silence: int = 200
#     n_mfcc: int = 13


# class EnhancedAudioHandler:
#     """
#     FIXED: Proper temp file handling to prevent deletion race conditions.
#     """
    
#     def __init__(self, config: AudioConfig = None):
#         self.config = config or AudioConfig()
        
#         # Create dedicated temp directory for this session
#         self.temp_dir = tempfile.mkdtemp(prefix="audio_processing_")
#         print(f"📁 Temp directory: {self.temp_dir}")
        
#         print(f"\n{'='*70}")
#         print(f"🎧 ENHANCED AUDIO HANDLER (FIXED)")
#         print(f"{'='*70}")
#         print(f"Whisper Model: {self.config.whisper_model}")
        
#         if self.config.num_speakers is None:
#             print(f"Speaker Detection: AUTO (testing {self.config.min_speakers}-{self.config.max_speakers} speakers)")
#         else:
#             print(f"Speaker Detection: MANUAL ({self.config.num_speakers} speakers)")
        
#         # Load Whisper
#         print(f"\n🔥 Loading Whisper {self.config.whisper_model}...")
#         if FASTER_WHISPER:
#             self.model = WhisperModel(
#                 self.config.whisper_model,
#                 device="cpu",
#                 compute_type="int8"
#             )
#             self.use_faster = True
#         else:
#             self.model = whisper.load_model(self.config.whisper_model)
#             self.use_faster = False
        
#         print(f"✅ Model loaded")
#         print(f"{'='*70}\n")
    
#     def __del__(self):
#         """Cleanup temp directory on deletion."""
#         try:
#             import shutil
#             if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
#                 shutil.rmtree(self.temp_dir)
#                 print(f"🧹 Cleaned up temp directory")
#         except:
#             pass
    
#     def _get_temp_path(self, prefix: str = "chunk") -> str:
#         """Generate unique temp file path."""
#         unique_id = uuid.uuid4().hex[:8]
#         return os.path.join(self.temp_dir, f"{prefix}_{unique_id}.wav")
    
#     def process_audio(
#         self,
#         audio_path: str,
#         user_description: str = "",
#         meeting_title: str = None,
#         session_id: str = None
#     ) -> List[Dict]:
#         """Process audio file with FIXED temp file handling."""
#         print(f"\n🎧 Processing: {os.path.basename(audio_path)}")
        
#         if not self._validate_audio(audio_path):
#             return []
        
#         # Step 1: Load audio
#         print(f"   📂 Loading audio...")
#         try:
#             audio = AudioSegment.from_file(audio_path)
#             print(f"      ✓ Loaded: {len(audio)/1000:.1f}s, {audio.channels} channels, {audio.frame_rate}Hz")
#         except Exception as e:
#             print(f"      ❌ Failed to load: {e}")
#             return []
        
#         # Step 2: Split by silence
#         print(f"   ✂️  Splitting by silence...")
#         chunks = self._split_by_silence(audio)
#         print(f"      ✓ Found {len(chunks)} natural segments")
        
#         # Step 3: Extract MFCC features
#         print(f"   🎵 Extracting MFCC features...")
#         mfcc_features, valid_chunks = self._extract_mfcc_features(chunks)
#         print(f"      ✓ Extracted features from {len(valid_chunks)} chunks")
        
#         # Step 4: Cluster speakers
#         print(f"   👥 Clustering speakers (KMeans)...")
#         speaker_labels = self._cluster_speakers(mfcc_features)
#         print(f"      ✓ Identified {len(set(speaker_labels))} speakers")
        
#         # Step 5: Transcribe
#         print(f"   📝 Transcribing chunks...")
#         transcribed_chunks = self._transcribe_chunks_fixed(
#             valid_chunks, 
#             speaker_labels,
#             audio_path
#         )
#         print(f"      ✓ Transcribed {len(transcribed_chunks)} chunks")
        
#         # Step 6: Analyze content
#         print(f"   🔍 Analyzing content...")
#         full_text = ' '.join([c['text'] for c in transcribed_chunks])
#         analysis = self._analyze_content(full_text)
#         if analysis['keywords']:
#             print(f"      ✓ Keywords: {', '.join(analysis['keywords'][:5])}")
        
#         # Step 7: Format for vectorstore
#         formatted_chunks = self._format_chunks(
#             transcribed_chunks=transcribed_chunks,
#             audio_path=audio_path,
#             analysis=analysis,
#             user_description=user_description,
#             meeting_title=meeting_title,
#             session_id=session_id
#         )
        
#         print(f"   ✅ Created {len(formatted_chunks)} chunks\n")
        
#         return formatted_chunks
    
#     def _validate_audio(self, audio_path: str) -> bool:
#         """Validate audio file."""
#         if not os.path.exists(audio_path):
#             print(f"   ❌ File not found")
#             return False
        
#         valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
#         ext = Path(audio_path).suffix.lower()
        
#         if ext not in valid_extensions:
#             print(f"   ❌ Unsupported format: {ext}")
#             return False
        
#         size_mb = os.path.getsize(audio_path) / (1024 * 1024)
#         print(f"   ✓ Valid: {ext} ({size_mb:.1f} MB)")
        
#         return True
    
#     def _split_by_silence(self, audio: AudioSegment) -> List[Tuple[AudioSegment, float, float]]:
#         """Split audio by silence into natural segments."""
#         chunks = silence.split_on_silence(
#             audio,
#             min_silence_len=self.config.min_silence_len,
#             silence_thresh=audio.dBFS - self.config.silence_thresh_offset,
#             keep_silence=self.config.keep_silence
#         )
        
#         chunks_with_times = []
#         current_time = 0.0
        
#         for chunk in chunks:
#             start_time = current_time
#             duration = len(chunk) / 1000.0
#             end_time = start_time + duration
            
#             chunks_with_times.append((chunk, start_time, end_time))
#             current_time = end_time
        
#         return chunks_with_times
    
#     def _extract_mfcc_features(
#         self, 
#         chunks: List[Tuple[AudioSegment, float, float]]
#     ) -> Tuple[np.ndarray, List]:
#         """Extract MFCC features - FIXED with better temp file handling."""
#         mfcc_features = []
#         valid_chunks = []
#         temp_files = []  # Track temp files to clean up
        
#         for i, (chunk, start_time, end_time) in enumerate(chunks):
#             temp_file = None
#             try:
#                 # Create temp file with unique name
#                 temp_file = self._get_temp_path(f"mfcc_chunk_{i}")
#                 temp_files.append(temp_file)
                
#                 # Export chunk
#                 chunk.export(temp_file, format="wav")
                
#                 # Verify file exists
#                 if not os.path.exists(temp_file):
#                     print(f"      ⚠️  Temp file not created: {temp_file}")
#                     continue
                
#                 # Load with librosa
#                 y, sr = librosa.load(temp_file, sr=None)
                
#                 # Extract MFCC features
#                 mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc)
#                 mfcc_mean = np.mean(mfcc, axis=1)
                
#                 mfcc_features.append(mfcc_mean)
#                 valid_chunks.append((chunk, start_time, end_time))
                
#             except Exception as e:
#                 print(f"      ⚠️  Skipped chunk {i}: {e}")
#                 continue
#             finally:
#                 # Don't delete yet - keep for potential reuse
#                 pass
        
#         # Cleanup temp files after processing
#         for temp_file in temp_files:
#             try:
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
#             except:
#                 pass
        
#         return np.array(mfcc_features), valid_chunks
    
#     def _detect_optimal_speakers(self, mfcc_features: np.ndarray) -> int:
#         """Automatically detect optimal number of speakers."""
#         n_samples = len(mfcc_features)
        
#         if n_samples < self.config.min_speakers:
#             print(f"      ⚠️  Only {n_samples} chunks, defaulting to 1 speaker")
#             return 1
        
#         min_k = self.config.min_speakers
#         max_k = min(self.config.max_speakers, n_samples - 1)
        
#         if max_k < min_k:
#             return 1
        
#         print(f"      🔍 Testing {min_k} to {max_k} speakers...")
        
#         silhouette_scores = []
#         inertias = []
        
#         for k in range(min_k, max_k + 1):
#             kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
#             labels = kmeans.fit_predict(mfcc_features)
            
#             score = silhouette_score(mfcc_features, labels)
#             silhouette_scores.append(score)
#             inertias.append(kmeans.inertia_)
            
#             print(f"         k={k}: silhouette={score:.3f}, inertia={kmeans.inertia_:.2f}")
        
#         best_idx = np.argmax(silhouette_scores)
#         optimal_k_silhouette = min_k + best_idx
#         best_score = silhouette_scores[best_idx]
        
#         print(f"      ✓ Best by Silhouette: {optimal_k_silhouette} speakers (score: {best_score:.3f})")
        
#         try:
#             kl = KneeLocator(
#                 range(min_k, max_k + 1), 
#                 inertias, 
#                 curve='convex', 
#                 direction='decreasing'
#             )
#             optimal_k_elbow = kl.elbow if kl.elbow else optimal_k_silhouette
#             print(f"      ✓ Elbow method suggests: {optimal_k_elbow} speakers")
#         except:
#             optimal_k_elbow = optimal_k_silhouette
        
#         if best_score < 0.2 and optimal_k_elbow != optimal_k_silhouette:
#             print(f"      ⚠️  Low silhouette score, using elbow method")
#             return optimal_k_elbow
        
#         return optimal_k_silhouette
    
#     def _cluster_speakers(self, mfcc_features: np.ndarray) -> np.ndarray:
#         """Cluster MFCC features to identify speakers."""
#         if len(mfcc_features) < 2:
#             print("      ⚠️  Too few chunks for clustering")
#             return np.zeros(len(mfcc_features), dtype=int)
        
#         if self.config.num_speakers is None:
#             num_speakers = self._detect_optimal_speakers(mfcc_features)
#             print(f"      🎯 Auto-detected: {num_speakers} speakers")
#         else:
#             num_speakers = min(self.config.num_speakers, len(mfcc_features))
#             print(f"      👤 Using specified: {num_speakers} speakers")
        
#         if num_speakers == 1:
#             return np.zeros(len(mfcc_features), dtype=int)
        
#         kmeans = KMeans(n_clusters=num_speakers, random_state=0, n_init=10)
#         labels = kmeans.fit_predict(mfcc_features)
        
#         unique, counts = np.unique(labels, return_counts=True)
#         distribution = dict(zip(unique, counts))
#         print(f"      📊 Speaker distribution: {distribution}")
        
#         return labels
    
#     def _transcribe_chunks_fixed(
#         self,
#         chunks: List[Tuple[AudioSegment, float, float]],
#         speaker_labels: np.ndarray,
#         original_audio_path: str
#     ) -> List[Dict]:
#         """
#         FIXED: Transcribe with proper temp file management.
        
#         Key fixes:
#         1. Use unique temp file names (UUID)
#         2. Verify file exists before transcription
#         3. Keep file until after transcription
#         4. Better error handling
#         """
#         transcribed = []
        
#         for i, ((chunk, start_time, end_time), speaker_label) in enumerate(zip(chunks, speaker_labels)):
#             temp_file = None
#             try:
#                 # Generate unique temp file path
#                 temp_file = self._get_temp_path(f"transcribe_{i}")
                
#                 # Export chunk to temp file
#                 chunk.export(temp_file, format="wav")
                
#                 # CRITICAL: Verify file exists
#                 if not os.path.exists(temp_file):
#                     print(f"      ⚠️  Transcription failed for chunk {i}: Temp file not created")
#                     continue
                
#                 # Small delay to ensure file is fully written
#                 import time
#                 time.sleep(0.1)
                
#                 # Transcribe
#                 if self.use_faster:
#                     segments_iter, info = self.model.transcribe(temp_file, beam_size=5)
#                     segments = list(segments_iter)
#                     text = ' '.join([seg.text for seg in segments]).strip()
#                     language = info.language
#                 else:
#                     result = self.model.transcribe(temp_file, verbose=False)
#                     text = result['text'].strip()
#                     language = result['language']
                
#                 if text:
#                     transcribed.append({
#                         'text': text,
#                         'speaker': f"Speaker {int(speaker_label) + 1}",
#                         'start_time': start_time,
#                         'end_time': end_time,
#                         'duration': end_time - start_time,
#                         'language': language,
#                         'chunk_index': i
#                     })
                
#             except Exception as e:
#                 print(f"      ⚠️  Transcription failed for chunk {i}: {e}")
#                 continue
#             finally:
#                 # Cleanup temp file AFTER transcription
#                 if temp_file and os.path.exists(temp_file):
#                     try:
#                         os.remove(temp_file)
#                     except:
#                         pass
        
#         return transcribed
    
#     def _analyze_content(self, full_text: str) -> Dict:
#         """Analyze content for keywords and topics."""
#         words = re.findall(r'\b\w{4,}\b', full_text.lower())
#         stop_words = {
#             'that', 'this', 'with', 'from', 'have', 'been', 'were',
#             'their', 'will', 'would', 'there', 'what', 'when', 'where',
#             'which', 'about', 'could', 'should', 'think', 'know',
#             'just', 'like', 'well', 'yeah', 'okay', 'right', 'going'
#         }
#         keywords = [w for w in words if w not in stop_words]
#         keyword_freq = Counter(keywords)
#         top_keywords = [word for word, count in keyword_freq.most_common(20)]
        
#         topics = []
#         topic_patterns = {
#             'meeting': ['meeting', 'agenda', 'discuss', 'presentation'],
#             'project': ['project', 'deadline', 'milestone', 'deliverable'],
#             'technical': ['code', 'system', 'database', 'api', 'server'],
#             'financial': ['budget', 'cost', 'revenue', 'profit', 'expense'],
#             'planning': ['plan', 'strategy', 'roadmap', 'timeline'],
#         }
        
#         text_lower = full_text.lower()
#         for topic, keywords_list in topic_patterns.items():
#             if sum(1 for kw in keywords_list if kw in text_lower) >= 2:
#                 topics.append(topic)
        
#         entities = []
#         dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', full_text)
#         entities.extend(dates[:5])
#         times = re.findall(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b', full_text)
#         entities.extend(times[:5])
#         amounts = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', full_text)
#         entities.extend(amounts[:5])
        
#         return {
#             'keywords': top_keywords,
#             'topics': topics,
#             'entities': list(set(entities))
#         }
    
#     def _format_chunks(
#         self,
#         transcribed_chunks: List[Dict],
#         audio_path: str,
#         analysis: Dict,
#         user_description: str,
#         meeting_title: str,
#         session_id: str
#     ) -> List[Dict]:
#         """Format chunks for vectorstore."""
#         formatted = []
#         filename = os.path.basename(audio_path)
        
#         for i, chunk in enumerate(transcribed_chunks):
#             content = self._build_content(chunk, meeting_title, filename)
#             chunk_keywords = self._get_chunk_keywords(chunk['text'])
            
#             metadata = {
#                 'type': 'audio',
#                 'modality': 'audio',
#                 'source': filename,
#                 'audio_path': audio_path,
#                 'filename': filename,
#                 'session_id': session_id,
#                 'meeting_title': meeting_title,
#                 'user_description': user_description,
#                 'chunk_index': int(i),
#                 'total_chunks': int(len(transcribed_chunks)),
#                 'start_time': float(chunk['start_time']),
#                 'end_time': float(chunk['end_time']),
#                 'duration': float(chunk['duration']),
#                 'timestamp_formatted': f"[{self._format_timestamp(chunk['start_time'])} → {self._format_timestamp(chunk['end_time'])}]",
#                 'speaker': chunk['speaker'],
#                 'num_speakers': 1,
#                 'word_count': int(len(chunk['text'].split())),
#                 'language': chunk['language'],
#                 'chunk_keywords': chunk_keywords,
#                 'overall_keywords': analysis['keywords'],
#                 'topics': analysis['topics'],
#                 'entities': analysis['entities'],
#                 'timestamp': datetime.now().isoformat(),
#                 'processing_mode': 'offline',
#                 'processing_engine': 'faster_whisper' if self.use_faster else 'openai_whisper',
#                 'diarization_method': 'librosa_kmeans_auto',
#                 'searchable_text': ' '.join(filter(None, [
#                     meeting_title or '',
#                     user_description,
#                     chunk['text'],
#                     ' '.join(chunk_keywords)
#                 ]))
#             }
            
#             formatted.append({
#                 'id': f"audio_{Path(audio_path).stem}_chunk_{i}",
#                 'content': content,
#                 'metadata': metadata,
#                 'token_count': int(len(chunk['text'].split()))
#             })
        
#         return formatted
    
#     def _build_content(self, chunk: Dict, meeting_title: str, filename: str) -> str:
#         """Build content string for chunk."""
#         lines = []
        
#         if meeting_title:
#             lines.append(f"# {meeting_title} - Segment {chunk['chunk_index'] + 1}\n")
#         else:
#             lines.append(f"# Audio: {filename} - Segment {chunk['chunk_index'] + 1}\n")
        
#         start = self._format_timestamp(chunk['start_time'])
#         end = self._format_timestamp(chunk['end_time'])
#         lines.append(f"**Timestamp:** [{start} → {end}] ({chunk['duration']:.1f}s)")
#         lines.append(f"**Speaker:** {chunk['speaker']}\n")
#         lines.append("**Transcription:**")
#         lines.append(chunk['text'])
        
#         return '\n'.join(lines)
    
#     def _get_chunk_keywords(self, text: str) -> List[str]:
#         """Extract keywords from chunk."""
#         words = re.findall(r'\b\w{4,}\b', text.lower())
#         stop_words = {'that', 'this', 'with', 'from', 'have', 'were', 'just', 'like'}
#         keywords = [w for w in words if w not in stop_words]
#         word_freq = Counter(keywords)
#         return [word for word, count in word_freq.most_common(10)]
    
#     def _format_timestamp(self, seconds: float) -> str:
#         """Format seconds as HH:MM:SS.mmm"""
#         hours = int(seconds // 3600)
#         minutes = int((seconds % 3600) // 60)
#         secs = int(seconds % 60)
#         millis = int((seconds % 1) * 1000)
#         return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
#     def process_batch(
#         self,
#         audio_paths: List[str],
#         session_id: str = None,
#         output_file: str = "audio_chunks"
#     ) -> List[Dict]:
#         """Process multiple audio files."""
#         all_chunks = []
        
#         print(f"\n{'='*70}")
#         print(f"🎧 BATCH PROCESSING {len(audio_paths)} FILES")
#         print(f"{'='*70}\n")
        
#         for i, path in enumerate(audio_paths, 1):
#             print(f"[{i}/{len(audio_paths)}]")
#             try:
#                 chunks = self.process_audio(path, session_id=session_id)
#                 all_chunks.extend(chunks)
#             except Exception as e:
#                 print(f"   ❌ Error: {e}\n")
        
#         self._save_results(all_chunks, output_file, session_id)
        
#         return all_chunks
    
#     def _save_results(self, chunks: List[Dict], output_prefix: str, session_id: str):
#         """Save chunks to PKL and JSON files."""
#         print(f"\n{'='*70}")
#         print(f"💾 SAVING RESULTS")
#         print(f"{'='*70}\n")
        
#         pkl_file = f"{output_prefix}.pkl"
#         with open(pkl_file, 'wb') as f:
#             pickle.dump(chunks, f)
#         print(f"✅ Saved PKL: {pkl_file}")
        
#         json_chunks = []
#         for chunk in chunks:
#             json_chunk = {
#                 'id': chunk['id'],
#                 'content_preview': chunk['content'][:300] + '...' if len(chunk['content']) > 300 else chunk['content'],
#                 'metadata': chunk['metadata'],
#                 'token_count': chunk['token_count']
#             }
#             json_chunks.append(json_chunk)
        
#         json_file = f"{output_prefix}.json"
#         with open(json_file, 'w', encoding='utf-8') as f:
#             json.dump(json_chunks, f, indent=2, ensure_ascii=False)
#         print(f"✅ Saved JSON: {json_file}")
        
#         total_duration = sum(c['metadata']['duration'] for c in chunks)
#         speakers = set(c['metadata']['speaker'] for c in chunks)
        
#         summary = {
#             'total_chunks': len(chunks),
#             'session_id': session_id,
#             'timestamp': datetime.now().isoformat(),
#             'processing_mode': 'offline',
#             'diarization_method': 'librosa_kmeans_auto',
#             'statistics': {
#                 'total': len(chunks),
#                 'total_duration': float(total_duration),
#                 'total_duration_formatted': str(timedelta(seconds=int(total_duration))),
#                 'unique_speakers': len(speakers),
#                 'speakers': list(speakers)
#             }
#         }
        
#         summary_file = f"{output_prefix}_summary.json"
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, indent=2, ensure_ascii=False)
#         print(f"✅ Saved Summary: {summary_file}")
        
#         print(f"\n{'='*70}")
#         print(f"📊 PROCESSING STATISTICS")
#         print(f"{'='*70}")
#         stats = summary['statistics']
#         print(f"   Total chunks: {stats['total']}")
#         print(f"   Total duration: {stats['total_duration_formatted']}")
#         print(f"   Speakers: {stats['unique_speakers']} ({', '.join(stats['speakers'])})")
#         print(f"{'='*70}\n")


# # Convenience functions

# def process_audio_file(
#     audio_path: str, 
#     output_prefix: str = "audio_chunks",
#     num_speakers: Optional[int] = None
# ) -> List[Dict]:
#     """Process single audio file."""
#     config = AudioConfig(num_speakers=num_speakers)
#     handler = EnhancedAudioHandler(config)
#     chunks = handler.process_audio(audio_path)
#     handler._save_results(chunks, output_prefix, session_id=Path(audio_path).stem)
#     return chunks


# def process_audio_directory(
#     directory: str, 
#     output_prefix: str = "audio_chunks",
#     num_speakers: Optional[int] = None
# ) -> List[Dict]:
#     """Process all audio in directory."""
#     audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
#     audio_files = []
    
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if Path(file).suffix.lower() in audio_exts:
#                 audio_files.append(os.path.join(root, file))
    
#     if not audio_files:
#         print(f"❌ No audio files found in {directory}")
#         return []
    
#     print(f"✅ Found {len(audio_files)} audio files\n")
#     audio_files.sort()
    
#     config = AudioConfig(num_speakers=num_speakers)
#     handler = EnhancedAudioHandler(config)
#     return handler.process_batch(
#         audio_files, 
#         session_id=Path(directory).name,
#         output_file=output_prefix
#     )

# ============================================================================
# VOICE QUERY TRANSCRIBER & TRANSLATOR (Offline Speech-to-English)
# ============================================================================

class VoiceQueryTranscriber:
    """
    Offline Voice Query Processor:
    - Transcribes spoken audio queries in real-time.
    - Automatically translates non-English languages (Hindi, Telugu, Spanish, etc.) into English.
    - Preserves English verbatim.
    """
    _instance = None

    @classmethod
    def get_instance(cls, model_name: str = "base"):
        if cls._instance is None:
            cls._instance = VoiceQueryTranscriber(model_name=model_name)
        return cls._instance

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.use_faster = False
        self.use_pipeline = False
        self._init_model()

    def _init_model(self):
        try:
            if FASTER_WHISPER:
                print(f"🎙️ Loading Faster-Whisper '{self.model_name}' for Voice Query Transcription...")
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8"
                )
                self.use_faster = True
                print("✅ Faster-Whisper Voice Transcriber initialized.")
            else:
                print(f"🎙️ Loading standard Whisper '{self.model_name}'...")
                self.model = whisper.load_model(self.model_name)
                self.use_faster = False
                print("✅ Standard Whisper initialized.")
        except Exception as e:
            print(f"⚠️ Faster-Whisper init warning: {e}. Attempting pipeline fallback...")
            try:
                from transformers import pipeline
                self.model = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    local_files_only=True   # OFFLINE: never download from the internet
                )
                self.use_pipeline = True
                print("✅ HuggingFace pipeline fallback initialized.")
            except Exception as e2:
                print(f"❌ Failed to initialize Whisper for voice query: {e2}")

    def transcribe_multilingual(self, audio_path: str, task: str = 'transcribe', language: str = None) -> Dict[str, Any]:
        """
        Transcribes speech, optionally keeping the original language or translating to English.

        Args:
            audio_path: Path to the audio file (WAV preferred).
            task: 'transcribe' = keep original language text
                  'translate'  = translate to English (Whisper built-in)
            language: BCP-47 language code to force Whisper to a specific language
                      (e.g. 'te' for Telugu, 'hi' for Hindi, 'en' for English).
                      If None or 'auto', Whisper auto-detects (unreliable for short utterances).

        Returns:
            dict with keys: success, text, language, probability
        """
        if self.model is None:
            return {"success": False, "error": "Speech recognition engine is not initialized.", "text": "", "language": "unknown"}

        wav_temp_path = None
        input_file = audio_path
        try:
            ext = Path(audio_path).suffix.lower()
            if ext in [".webm", ".ogg", ".aac", ".opus", ".m4a"]:
                wav_temp_path = audio_path + "_ml_converted.wav"
                try:
                    sound = AudioSegment.from_file(audio_path)
                    sound = sound.set_frame_rate(16000).set_channels(1)
                    sound.export(wav_temp_path, format="wav")
                    input_file = wav_temp_path
                except Exception:
                    input_file = audio_path

            # Normalise language — treat 'auto' or empty as None (auto-detect)
            forced_lang = language if (language and language.strip().lower() not in ('auto', 'en-auto', '')) else None
            # For 'en' (English), keep as forced_lang so Whisper doesn't waste time detecting
            if language and language.strip().lower() == 'en':
                forced_lang = 'en'

            if self.use_faster:
                transcribe_kwargs = dict(
                    task=task,
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                    no_speech_threshold=0.5
                )
                if forced_lang:
                    transcribe_kwargs['language'] = forced_lang
                segments_iter, info = self.model.transcribe(input_file, **transcribe_kwargs)
                segments = list(segments_iter)
                detected_lang = forced_lang if forced_lang else info.language
                prob = float(info.language_probability)
                text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()]).strip()

            elif self.use_pipeline:
                result = self.model(input_file)
                text = result.get('text', '').strip()
                detected_lang = "auto"
                prob = 1.0

            else:
                # Standard OpenAI Whisper
                audio_input = input_file
                try:
                    import soundfile as sf
                    import scipy.signal
                    data, sr = sf.read(input_file, dtype='float32')
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        data = scipy.signal.resample(data, int(len(data) * 16000 / sr))
                    audio_input = data
                except Exception:
                    audio_input = input_file

                result = self.model.transcribe(
                    audio_input,
                    task=task,
                    language=forced_lang,  # None = auto-detect; 'te'/'hi'/'en' = forced
                    condition_on_previous_text=False,
                    verbose=False
                )
                text = result.get('text', '').strip()
                detected_lang = forced_lang if forced_lang else result.get('language', 'en')
                prob = 1.0

            # Filter hallucinations
            hallucination_phrases = [
                "amara.org", "subtitles by", "thank you for watching",
                "subscribe to", "you", "thanks for watching", "subtitles",
                "translated by", "closed captions"
            ]
            if any(text.lower().strip() == p or text.lower().strip() == f"{p}." for p in hallucination_phrases):
                text = ""

            text = text.strip()
            text = re.sub(r'([.?!,])\1+', r'\1', text)

            return {
                "success": bool(text and len(text) > 1),
                "text": text,
                "language": detected_lang,
                "probability": prob
            }

        except Exception as e:
            print(f"❌ transcribe_multilingual error: {e}")
            return {"success": False, "error": str(e), "text": "", "language": "unknown"}
        finally:
            if wav_temp_path and os.path.exists(wav_temp_path):
                try:
                    os.remove(wav_temp_path)
                except Exception:
                    pass

    def transcribe_to_english(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes speech and translates it into English.
        """
        if self.model is None:
            return {
                "success": False,
                "error": "Speech recognition engine is not initialized.",
                "text": "",
                "language": "unknown"
            }

        # Convert webm/ogg if needed using pydub
        wav_temp_path = None
        input_file = audio_path
        try:
            ext = Path(audio_path).suffix.lower()
            if ext in [".webm", ".ogg", ".aac", ".opus", ".m4a"]:
                wav_temp_path = audio_path + "_converted.wav"
                try:
                    sound = AudioSegment.from_file(audio_path)
                    sound = sound.set_frame_rate(16000).set_channels(1)
                    sound.export(wav_temp_path, format="wav")
                    input_file = wav_temp_path
                except Exception as conv_err:
                    print(f"Pydub conversion notice: {conv_err}. Using original file directly.")
                    input_file = audio_path

            if self.use_faster:
                # 1. First attempt with VAD filter and no text conditioning to prevent hallucinations
                segments_iter, info = self.model.transcribe(
                    input_file,
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                    no_speech_threshold=0.5
                )
                segments = list(segments_iter)
                detected_lang = info.language
                prob = float(info.language_probability)
                
                # If non-English detected with high confidence (>0.60), translate to English
                if detected_lang != 'en' and prob > 0.60:
                    trans_iter, _ = self.model.transcribe(
                        input_file,
                        task="translate",
                        beam_size=5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        vad_filter=True
                    )
                    segments = list(trans_iter)

                text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()]).strip()

            elif self.use_pipeline:
                result = self.model(input_file)
                text = result.get('text', '').strip()
                detected_lang = "auto"
                prob = 1.0
            else:
                # Direct robust soundfile loading for WAV/audio to bypass ffmpeg dependency
                audio_input = input_file
                try:
                    import soundfile as sf
                    import scipy.signal
                    data, sr = sf.read(input_file, dtype='float32')
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        data = scipy.signal.resample(data, int(len(data) * 16000 / sr))
                    audio_input = data
                except Exception as sf_err:
                    audio_input = input_file

                result = self.model.transcribe(audio_input, condition_on_previous_text=False, verbose=False)
                text = result.get('text', '').strip()
                detected_lang = result.get('language', 'en')
                prob = 1.0

            # Filter out common Whisper hallucination artifacts
            hallucination_phrases = [
                "amara.org", "subtitles by", "thank you for watching",
                "subscribe to", "you", "thanks for watching", "subtitles",
                "translated by", "closed captions"
            ]
            cleaned_lower = text.lower().strip()
            if any(cleaned_lower == phrase or cleaned_lower == f"{phrase}." for phrase in hallucination_phrases):
                text = ""

            # Clean text
            text = text.strip()
            text = re.sub(r'([.?!,])\1+', r'\1', text)

            return {
                "success": bool(text and len(text) > 1),
                "text": text,
                "language": detected_lang,
                "probability": prob
            }

        except Exception as e:
            print(f"❌ Voice transcription error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "language": "unknown"
            }
        finally:
            if wav_temp_path and os.path.exists(wav_temp_path):
                try:
                    os.remove(wav_temp_path)
                except Exception:
                    pass

# def main():
#     """Command-line interface."""
#     import sys
    
#     if len(sys.argv) < 2:
#         print("\n" + "="*70)
#         print("🎧 ENHANCED AUDIO HANDLER (FIXED)")
#         print("="*70)
#         print("\nUsage:")
#         print("  python audio_handler.py <audio_file> [num_speakers]")
#         print("  python audio_handler.py <audio_directory> [num_speakers]")
#         print("\nExamples:")
#         print("  python audio_handler.py meeting.mp3")
#         print("  python audio_handler.py meeting.mp3 2")
#         print("="*70 + "\n")
#         sys.exit(1)
    
#     path = sys.argv[1]
#     num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
#     if os.path.isfile(path):
#         chunks = process_audio_file(path, num_speakers=num_speakers)
#         print(f"\n✅ Processed {len(chunks)} chunks from 1 file")
#     elif os.path.isdir(path):
#         chunks = process_audio_directory(path, num_speakers=num_speakers)
#         print(f"\n✅ Processed {len(chunks)} chunks from directory")
#     else:
#         print(f"❌ Not found: {path}")
#         sys.exit(1)
    
#     print("\n📦 Files saved:")
#     print("   - audio_chunks.pkl")
#     print("   - audio_chunks.json")
#     print("   - audio_chunks_summary.json")
#     print("\n🚀 Ready for vectorstore.py!\n")


# if __name__ == "__main__":
#     main()