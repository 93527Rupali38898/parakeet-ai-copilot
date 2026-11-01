"""
Speech-to-Text Engine for Parakeet AI Prototype.
Provides local offline transcription using faster-whisper (tiny/base models),
running 100% locally on CPU or GPU with int8 quantization.

Author: Rupali Goyal
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    WhisperModel = None
    HAS_FASTER_WHISPER = False


class STTEngine:
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the local Speech-to-Text engine.
        model_size: "tiny" (~39MB, fastest), "base" (~75MB), or "small".
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[Any] = None
        self._is_loading = False

    def load_model(self) -> bool:
        """Loads the local faster-whisper model into memory."""
        if self.model is not None:
            return True
        if not HAS_FASTER_WHISPER:
            return False

        try:
            self._is_loading = True
            # Load locally cached faster-whisper model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self._is_loading = False
            return True
        except Exception as e:
            self._is_loading = False
            print(f"[STT Error] Failed to load model: {e}")
            return False

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, language: str = "en") -> str:
        """
        Transcribes audio numpy array (1D float32 normalized between -1.0 and 1.0).
        """
        if len(audio_data) == 0:
            return ""

        # Minimum duration check (at least 0.3s of audio)
        min_samples = int(sample_rate * 0.3)
        if len(audio_data) < min_samples:
            return ""

        # Check for absolute silence
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 0.005:
            # Virtually complete silence
            return ""

        if self.model is None:
            success = self.load_model()
            if not success or self.model is None:
                raise RuntimeError("Speech-to-Text engine is not loaded. Please ensure faster-whisper is installed.")

        try:
            # faster-whisper takes 16kHz float32 array directly
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=3,
                language=language,
                vad_filter=True,  # Voice activity detection to trim silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            return transcribed_text
        except Exception as e:
            print(f"[STT Error] Transcription failed: {e}")
            raise e

    def transcribe_file(self, file_path: str, language: str = "en") -> str:
        """Transcribes audio from a WAV/MP3 file on disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if self.model is None:
            self.load_model()

        segments, info = self.model.transcribe(
            file_path,
            beam_size=3,
            language=language,
            vad_filter=True
        )
        return " ".join([segment.text for segment in segments]).strip()


# Preset test questions for demo and testing without microphone
SAMPLE_QUESTIONS: List[Dict[str, str]] = [
    {
        "title": "Interview Question (Technical)",
        "question": "Can you explain the difference between supervised and unsupervised learning, with practical real-world examples?"
    },
    {
        "title": "Architecture Question (System Design)",
        "question": "What are microservices, and what are the main tradeoffs compared to a monolithic architecture?"
    },
    {
        "title": "AI / NLP Concept",
        "question": "How does the self-attention mechanism work in a Transformer model?"
    },
    {
        "title": "Behavioral Interview",
        "question": "Tell me about a challenging technical bug you encountered and how you systematically diagnosed and resolved it."
    },
    {
        "title": "General Knowledge",
        "question": "What is the theory of relativity and why is it important in modern physics?"
    }
]

