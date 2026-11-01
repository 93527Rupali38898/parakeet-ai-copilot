"""
Unit and Integration Tests for Parakeet AI Prototype.
Verifies STT transcription, audio recording primitives, and streaming LLM generation.

Author: Rupali Goyal
"""

import time
import unittest
import numpy as np
from audio_recorder import AudioRecorder
from stt_engine import STTEngine, SAMPLE_QUESTIONS
from llm_engine import LLMEngine, OfflineFastBackend, OllamaBackend


class TestParakeetPrototype(unittest.TestCase):
    def test_audio_recorder_devices(self):
        """Verify device discovery does not throw and returns a list."""
        devices = AudioRecorder.get_input_devices()
        self.assertIsInstance(devices, list)

    def test_offline_llm_streaming(self):
        """Verify progressive token streaming from offline local engine."""
        engine = OfflineFastBackend()
        tokens = []
        for token in engine.stream_generate("Explain supervised vs unsupervised learning"):
            tokens.append(token)
            if len(tokens) >= 10:
                break

        full_text = "".join(tokens)
        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(full_text.strip()), 0)
        print(f"[TEST PASS] Streamed {len(tokens)} tokens: {full_text[:50]}...")

    def test_llm_engine_controller(self):
        """Test unified LLM controller with callback queue."""
        controller = LLMEngine(backend_type="offline")
        collected_tokens = []
        completed = False

        def on_token(t):
            collected_tokens.append(t)

        def on_complete():
            nonlocal completed
            completed = True

        def on_error(err):
            pass

        controller.start_streaming(
            prompt="What is self-attention?",
            on_token=on_token,
            on_complete=on_complete,
            on_error=on_error,
            backend_name="offline"
        )

        # Wait up to 3 seconds
        start = time.time()
        while time.time() - start < 3.0 and not completed:
            time.sleep(0.05)

        self.assertGreater(len(collected_tokens), 0)
        print(f"[TEST PASS] Controller streamed {len(collected_tokens)} tokens.")

    def test_stt_engine_synthetic_audio(self):
        """Test STT transcription with silence/synthetic signal."""
        stt = STTEngine(model_size="tiny", device="cpu", compute_type="int8")
        stt.load_model()
        self.assertIsNotNone(stt.model)

        # 0.5 seconds of synthetic audio
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)) * 0.05).astype(np.float32)
        result = stt.transcribe_audio(samples, sample_rate=16000)
        self.assertIsInstance(result, str)
        print(f"[TEST PASS] STT transcription ran smoothly.")


if __name__ == "__main__":
    unittest.main()
