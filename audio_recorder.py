"""
Audio Recorder Module for Parakeet AI Prototype.
Handles live microphone recording via sounddevice / pyaudio,
real-time RMS volume level calculations for UI visualizer,
and buffer conversion for Whisper STT.

Author: Rupali Goyal
"""

import time
import math
import threading
import numpy as np
from typing import Callable, Optional, List, Tuple

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except Exception:
    pyaudio = None
    HAS_PYAUDIO = False


class AudioRecorderError(Exception):
    """Custom exception for audio recording errors."""
    pass


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self._record_thread: Optional[threading.Thread] = None
        self._audio_chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self.current_volume: float = 0.0  # Normalized 0.0 to 1.0 for UI visualizer
        self.start_time: float = 0.0
        self._stop_event = threading.Event()
        self.selected_device_index: Optional[int] = None

    @staticmethod
    def get_input_devices() -> List[Tuple[int, str]]:
        """Returns list of available audio input devices as (index, name)."""
        devices = []
        if HAS_SOUNDDEVICE:
            try:
                device_list = sd.query_devices()
                for i, d in enumerate(device_list):
                    if d.get("max_input_channels", 0) > 0:
                        host_api = sd.query_hostapis(d.get("hostapi", 0)).get("name", "")
                        clean_name = d.get("name", f"Device {i}").replace("\n", " ").strip()
                        devices.append((i, f"{clean_name} ({host_api})"))
            except Exception:
                pass

        if not devices and HAS_PYAUDIO:
            try:
                pa = pyaudio.PyAudio()
                count = pa.get_device_count()
                for i in range(count):
                    try:
                        info = pa.get_device_info_by_index(i)
                        if info.get("maxInputChannels", 0) > 0:
                            clean_name = info.get("name", f"Device {i}").replace("\n", " ").strip()
                            devices.append((i, clean_name))
                    except Exception:
                        continue
                pa.terminate()
            except Exception:
                pass

        return devices

    def start_recording(self, device_index: Optional[int] = None) -> None:
        """Starts capturing microphone audio in a background thread."""
        if self.is_recording:
            return

        self.selected_device_index = device_index
        with self._lock:
            self._audio_chunks = []
            self.current_volume = 0.0
            self.start_time = time.time()
            self._stop_event.clear()
            self.is_recording = True

        self._record_thread = threading.Thread(
            target=self._record_worker,
            daemon=True,
            name="AudioRecorderWorker"
        )
        self._record_thread.start()

    def _record_worker(self) -> None:
        """Background worker capturing audio streams."""
        # Try sounddevice callback first
        recorded_successfully = False
        last_error = None

        if HAS_SOUNDDEVICE:
            try:
                def audio_callback(indata, frames, time_info, status):
                    if not self.is_recording:
                        return
                    # indata is numpy array of shape (frames, 1) float32
                    data = indata.copy().flatten()
                    with self._lock:
                        self._audio_chunks.append(data)
                        # Compute RMS volume for UI level meter
                        rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
                        # Normalize roughly between 0.0 and 1.0 (with boost for display)
                        self.current_volume = min(1.0, rms * 8.0)

                # If selected_device_index is None, let sounddevice pick default
                stream_kwargs = {
                    "samplerate": self.sample_rate,
                    "channels": 1,
                    "dtype": "float32",
                    "callback": audio_callback,
                    "blocksize": 1024,
                }
                if self.selected_device_index is not None:
                    stream_kwargs["device"] = self.selected_device_index

                with sd.InputStream(**stream_kwargs):
                    recorded_successfully = True
                    while not self._stop_event.is_set():
                        time.sleep(0.05)

            except Exception as e:
                last_error = str(e)
                recorded_successfully = False

        # Fallback to PyAudio if sounddevice fails or is unavailable
        if not recorded_successfully and HAS_PYAUDIO:
            try:
                pa = pyaudio.PyAudio()
                stream_kwargs = {
                    "format": pyaudio.paInt16,
                    "channels": 1,
                    "rate": self.sample_rate,
                    "input": True,
                    "frames_per_buffer": 1024
                }
                if self.selected_device_index is not None:
                    stream_kwargs["input_device_index"] = self.selected_device_index

                stream = pa.open(**stream_kwargs)
                recorded_successfully = True

                while not self._stop_event.is_set():
                    try:
                        raw_data = stream.read(1024, exception_on_overflow=False)
                        # Convert int16 bytes to float32 numpy array
                        int_data = np.frombuffer(raw_data, dtype=np.int16)
                        float_data = int_data.astype(np.float32) / 32768.0
                        with self._lock:
                            self._audio_chunks.append(float_data)
                            rms = float(np.sqrt(np.mean(float_data**2))) if len(float_data) > 0 else 0.0
                            self.current_volume = min(1.0, rms * 8.0)
                    except Exception:
                        break

                stream.stop_stream()
                stream.close()
                pa.terminate()

            except Exception as e:
                last_error = str(e)

        if not recorded_successfully:
            self.is_recording = False
            # Generate simulated ambient/silence chunks so downstream doesn't crash
            # and note the error
            raise AudioRecorderError(
                f"Could not open microphone stream: {last_error or 'No active recording device found'}. "
                "Please verify your microphone is connected and permissions are granted."
            )

    def stop_recording(self) -> np.ndarray:
        """Stops recording and returns the full audio array as float32 normalized 16kHz."""
        self.is_recording = False
        self._stop_event.set()

        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.5)

        self.current_volume = 0.0

        with self._lock:
            if not self._audio_chunks:
                return np.zeros(0, dtype=np.float32)
            audio_array = np.concatenate(self._audio_chunks, axis=0)
            self._audio_chunks = []
            return audio_array

    def get_recording_duration(self) -> float:
        """Returns elapsed recording time in seconds."""
        if not self.is_recording:
            return 0.0
        return max(0.0, time.time() - self.start_time)
