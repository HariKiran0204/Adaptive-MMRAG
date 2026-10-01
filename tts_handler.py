"""
tts_handler.py - Offline Text-to-Speech (Audio Generation) Engine for Adaptive-MMRAG.

Features:
1. 100% Offline execution using local speech synthesis (Windows SAPI5 / pyttsx3).
2. Natural text sanitization (strips markdown, code blocks, citations, HTML tags).
3. Thread-safe COM initialization for Flask multi-threaded request handling.
4. Voice selection (David / Zira / system default).
5. Audio file caching and cleanup.
"""

import os
import re
import time
import uuid
import hashlib
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class OfflineTTSHandler:
    """
    Offline Text-to-Speech engine providing natural speech synthesis
    for Adaptive-MMRAG responses.
    """

    def __init__(self, cache_dir: str = "audio_cache", speech_rate: int = 165):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.speech_rate = speech_rate
        self._lock = threading.Lock()
        self.available_voices = self._detect_voices()

    def _detect_voices(self) -> Dict[str, str]:
        """Detect locally installed TTS voices."""
        voices_map = {}
        if not PYTTSX3_AVAILABLE:
            return voices_map

        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            engine = pyttsx3.init()
            installed = engine.getProperty('voices')
            for v in installed:
                name_lower = v.name.lower()
                if 'zira' in name_lower:
                    voices_map['zira'] = v.id
                    voices_map['woman'] = v.id
                    voices_map['female'] = v.id
                elif 'david' in name_lower:
                    voices_map['david'] = v.id
                    voices_map['man'] = v.id
                    voices_map['male'] = v.id
                else:
                    voices_map[v.name] = v.id
            if installed and 'default' not in voices_map:
                voices_map['default'] = installed[0].id
        except Exception as e:
            print(f"[TTS][WARN] Voice detection notice: {e}")

        return voices_map

    def clean_text_for_speech(self, text: str) -> str:
        """
        Sanitizes assistant responses for natural, listenable speech:
        Removes markdown, tags, citations, tables, etc.
        """
        if not text:
            return ""

        clean = re.sub(r'<details.*?</details>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        clean = re.sub(r'```.*?```', ' ', clean, flags=re.DOTALL)
        clean = re.sub(r'`[^`]+`', ' ', clean)
        clean = re.sub(r'\|.*?\|', ' ', clean)
        clean = re.sub(r'\[Source:[^\]]+\]', ' ', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\[\d+\]', ' ', clean)
        clean = re.sub(r'>\s*[^a-zA-Z0-9\s]+\s*\[[^\]]+\]', ' ', clean)
        clean = re.sub(r'[*#_~]+', ' ', clean)
        clean = re.sub(r'https?://\S+', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def synthesize(
        self,
        text: str,
        voice_preference: str = "default",
        rate: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Synthesize text to an offline WAV audio file.

        Returns:
            (success: bool, audio_filename_or_err: str, cleaned_text: str)
        """
        if not PYTTSX3_AVAILABLE:
            return False, "pyttsx3 library is not installed", ""

        cleaned = self.clean_text_for_speech(text)
        if not cleaned:
            return False, "No speakable text found in response", ""

        if len(cleaned) > 2000:
            cleaned = cleaned[:2000] + "... and more details are available in the document."

        rate_val = rate or self.speech_rate
        hash_sig = hashlib.md5(f"{cleaned}_{voice_preference}_{rate_val}".encode('utf-8')).hexdigest()[:12]
        out_filename = f"speech_{hash_sig}.wav"
        out_path = self.cache_dir / out_filename

        if out_path.exists() and out_path.stat().st_size > 1000:
            return True, out_filename, cleaned

        with self._lock:
            try:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except Exception:
                    pass

                engine = pyttsx3.init()
                engine.setProperty('rate', rate_val)
                engine.setProperty('volume', 1.0)

                v_pref = voice_preference.lower()
                selected_voice_id = (
                    self.available_voices.get(v_pref)
                    or self.available_voices.get('default')
                )
                if selected_voice_id:
                    engine.setProperty('voice', selected_voice_id)

                engine.save_to_file(cleaned, str(out_path))
                engine.runAndWait()

                if out_path.exists() and out_path.stat().st_size > 500:
                    return True, out_filename, cleaned
                else:
                    return False, "Generated audio file was empty", cleaned

            except Exception as e:
                print(f"[TTS][ERROR] Speech synthesis failed: {e}")
                return False, str(e), cleaned


tts_engine = OfflineTTSHandler()
