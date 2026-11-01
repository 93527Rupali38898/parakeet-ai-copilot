"""
Desktop GUI for Parakeet AI Prototype.
Built with CustomTkinter for a modern, sleek interface inspired by Parakeet AI.
Provides Start/Stop microphone recording, live audio visualizer, Whisper STT,
and progressive streaming LLM answer generation.

Author: Rupali Goyal
"""

import time
import queue
import threading
import tkinter as tk
from typing import Optional, List, Dict

import customtkinter as ctk

from audio_recorder import AudioRecorder, AudioRecorderError
from stt_engine import STTEngine, SAMPLE_QUESTIONS
from llm_engine import LLMEngine, OllamaBackend

# Configure CustomTkinter aesthetic
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class ParakeetDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Parakeet AI — Desktop Prototype")
        self.geometry("900x780")
        self.minsize(780, 680)

        # Core Engines
        self.recorder = AudioRecorder(sample_rate=16000)
        self.stt = STTEngine(model_size="tiny", device="cpu", compute_type="int8")
        self.llm = LLMEngine(backend_type="offline")

        # Application state
        self._sample_index = 0
        self._poll_timer_id = None
        self._stream_token_count = 0
        self._stream_start_time = 0.0
        self._ui_queue = queue.Queue()

        # Build UI layout
        self._setup_ui()

        # Start UI event polling loop
        self._process_ui_queue()

        # Preload STT model in background so UI opens instantly
        threading.Thread(target=self._preload_stt, daemon=True).start()

        # Populate audio devices & Ollama status in background
        threading.Thread(target=self._detect_system_devices, daemon=True).start()

    def _preload_stt(self):
        """Preloads faster-whisper model into memory silently."""
        try:
            self.stt.load_model()
            self._ui_queue.put(lambda: self._set_status("Speech engine ready (faster-whisper tiny)"))
        except Exception as e:
            self._ui_queue.put(lambda: self._set_status(f"Speech engine initialization warning: {e}"))

    def _detect_system_devices(self):
        """Scans for available microphones and local Ollama server."""
        devices = self.recorder.get_input_devices()
        ollama_running = self.llm.ollama.is_available()
        ollama_models = self.llm.ollama.get_installed_models() if ollama_running else []

        def update():
            # Update Mic selector
            device_names = [f"[{idx}] {name[:35]}" for idx, name in devices]
            if not device_names:
                device_names = ["Default System Microphone"]
            self.mic_dropdown.configure(values=device_names)
            self.mic_dropdown.set(device_names[0])

            # Update LLM Backend
            if ollama_running:
                self.backend_dropdown.set("Ollama (Local Daemon)")
                self.llm_status_badge.configure(
                    text="● Ollama Connected",
                    text_color="#10b981"
                )
                if ollama_models:
                    self.model_dropdown.configure(values=ollama_models)
                    self.model_dropdown.set(ollama_models[0])
            else:
                self.backend_dropdown.set("Offline Fast Engine (Instant)")
                self.llm_status_badge.configure(
                    text="○ Ollama Offline (Using Fast Local Engine)",
                    text_color="#f59e0b"
                )

        self._ui_queue.put(update)

    def _setup_ui(self):
        """Constructs the sleek Parakeet AI desktop interface."""
        # Top Header Frame
        header = ctk.CTkFrame(self, corner_radius=12, fg_color=("#e2e8f0", "#18181b"))
        header.pack(fill="x", padx=16, pady=(14, 8))

        # Title & Subtitle
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=12)

        title_lbl = ctk.CTkLabel(
            title_box,
            text="🦜 Parakeet AI",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ctk.CTkLabel(
            title_box,
            text="Desktop Prototype • Local Speech-to-Text & Progressive AI Answer",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray65")
        )
        subtitle_lbl.pack(anchor="w")

        # Top Right Controls (Theme & System Badges)
        right_header = ctk.CTkFrame(header, fg_color="transparent")
        right_header.pack(side="right", padx=16, pady=12)

        self.llm_status_badge = ctk.CTkLabel(
            right_header,
            text="Checking local LLM...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray60"
        )
        self.llm_status_badge.pack(side="top", anchor="e", pady=(0, 4))

        theme_switch = ctk.CTkSwitch(
            right_header,
            text="Dark Mode",
            command=self._toggle_theme,
            font=ctk.CTkFont(size=11)
        )
        theme_switch.select()
        theme_switch.pack(side="top", anchor="e")

        # Main Scrollable / Stacked Content
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=16, pady=4)

        # -------------------------------------------------------------
        # CARD 1: Audio Capture & Transcription Section
        # -------------------------------------------------------------
        audio_card = ctk.CTkFrame(main_frame, corner_radius=10)
        audio_card.pack(fill="x", pady=6)

        # Card Title Bar with Controls
        bar1 = ctk.CTkFrame(audio_card, fg_color="transparent")
        bar1.pack(fill="x", padx=14, pady=(12, 6))

        sec1_title = ctk.CTkLabel(
            bar1,
            text="🎙️ Microphone Audio Capture",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        sec1_title.pack(side="left")

        # Device selector dropdown
        self.mic_dropdown = ctk.CTkOptionMenu(
            bar1,
            values=["Default Microphone"],
            width=210,
            height=28,
            font=ctk.CTkFont(size=11)
        )
        self.mic_dropdown.pack(side="right")

        device_lbl = ctk.CTkLabel(bar1, text="Input:", font=ctk.CTkFont(size=11), text_color="gray60")
        device_lbl.pack(side="right", padx=(0, 6))

        # Big Buttons & Recording Visualizer Bar
        btn_bar = ctk.CTkFrame(audio_card, fg_color="transparent")
        btn_bar.pack(fill="x", padx=14, pady=6)

        self.btn_start = ctk.CTkButton(
            btn_bar,
            text="▶  Start Recording",
            command=self._on_start_recording,
            fg_color="#10b981",
            hover_color="#059669",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=160,
            height=36
        )
        self.btn_start.pack(side="left", padx=(0, 10))

        self.btn_stop = ctk.CTkButton(
            btn_bar,
            text="⏹  Stop Recording",
            command=self._on_stop_recording,
            fg_color="#ef4444",
            hover_color="#dc2626",
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled",
            width=160,
            height=36
        )
        self.btn_stop.pack(side="left", padx=(0, 14))

        # Recording timer and live volume bar
        self.rec_status_lbl = ctk.CTkLabel(
            btn_bar,
            text="Idle • Ready",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray65"
        )
        self.rec_status_lbl.pack(side="left", padx=(0, 12))

        self.volume_bar = ctk.CTkProgressBar(btn_bar, width=140, height=12)
        self.volume_bar.set(0.0)
        self.volume_bar.pack(side="left", padx=(0, 10))

        # Transcription Output Area
        transcription_header = ctk.CTkFrame(audio_card, fg_color="transparent")
        transcription_header.pack(fill="x", padx=14, pady=(8, 2))

        lbl_trans = ctk.CTkLabel(
            transcription_header,
            text="Transcribed Question / Prompt (Editable):",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_trans.pack(side="left")

        # Action pills
        btn_sample = ctk.CTkButton(
            transcription_header,
            text="💡 Load Sample Question",
            command=self._load_next_sample_question,
            width=150,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90")
        )
        btn_sample.pack(side="right")

        btn_clear = ctk.CTkButton(
            transcription_header,
            text="🗑️ Clear",
            command=self._clear_transcription,
            width=65,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90")
        )
        btn_clear.pack(side="right", padx=6)

        # Text area for transcription
        self.txt_transcription = ctk.CTkTextbox(
            audio_card,
            height=75,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            wrap="word"
        )
        self.txt_transcription.pack(fill="x", padx=14, pady=(2, 12))
        self.txt_transcription.insert(
            "1.0",
            "Click 'Start Recording' and speak your question into the microphone, "
            "or click 'Load Sample Question' to test without speaking."
        )

        # -------------------------------------------------------------
        # CARD 2: AI Answer Suggestion Section (Streaming)
        # -------------------------------------------------------------
        ai_card = ctk.CTkFrame(main_frame, corner_radius=10)
        ai_card.pack(fill="both", expand=True, pady=6)

        # Controls row
        ai_controls = ctk.CTkFrame(ai_card, fg_color="transparent")
        ai_controls.pack(fill="x", padx=14, pady=(12, 6))

        # Primary Suggest Answer Button
        self.btn_suggest = ctk.CTkButton(
            ai_controls,
            text="✨  Suggest Answer",
            command=self._on_suggest_answer,
            fg_color="#3b82f6",
            hover_color="#2563eb",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=170,
            height=36
        )
        self.btn_suggest.pack(side="left", padx=(0, 10))

        self.btn_stop_ai = ctk.CTkButton(
            ai_controls,
            text="⏹ Stop",
            command=self._on_stop_generation,
            fg_color="#71717a",
            hover_color="#52525b",
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled",
            width=80,
            height=36
        )
        self.btn_stop_ai.pack(side="left", padx=(0, 14))

        # Backend selector
        self.backend_dropdown = ctk.CTkOptionMenu(
            ai_controls,
            values=[
                "Offline Fast Engine (Instant)",
                "Ollama (Local Daemon)",
                "Local Transformers"
            ],
            command=self._on_backend_changed,
            width=190,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        self.backend_dropdown.pack(side="right")

        backend_lbl = ctk.CTkLabel(ai_controls, text="LLM:", font=ctk.CTkFont(size=11), text_color="gray60")
        backend_lbl.pack(side="right", padx=(0, 6))

        # Model name selector
        self.model_dropdown = ctk.CTkOptionMenu(
            ai_controls,
            values=["llama3.2", "llama3.2:1b", "mistral", "phi3"],
            width=120,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        self.model_dropdown.pack(side="right", padx=(0, 10))

        # Answer Label & Copy
        answer_bar = ctk.CTkFrame(ai_card, fg_color="transparent")
        answer_bar.pack(fill="x", padx=14, pady=(6, 2))

        lbl_answer = ctk.CTkLabel(
            answer_bar,
            text="AI Suggested Answer (Progressive Streaming):",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_answer.pack(side="left")

        self.lbl_stats = ctk.CTkLabel(
            answer_bar,
            text="Waiting for prompt...",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.lbl_stats.pack(side="left", padx=14)

        btn_copy = ctk.CTkButton(
            answer_bar,
            text="📋 Copy Answer",
            command=self._copy_answer_to_clipboard,
            width=110,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90")
        )
        btn_copy.pack(side="right")

        # Answer Rich Output Area (Progressive token streaming display)
        self.txt_answer = ctk.CTkTextbox(
            ai_card,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word"
        )
        self.txt_answer.pack(fill="both", expand=True, padx=14, pady=(2, 12))
        self.txt_answer.insert(
            "1.0",
            "Your AI response will stream here progressively token-by-token...\n\n"
            "• Connect your microphone and click 'Start Recording'.\n"
            "• Speak your question and click 'Stop Recording' to transcribe.\n"
            "• Click 'Suggest Answer' to stream the local AI answer!"
        )

        # -------------------------------------------------------------
        # Bottom Status Bar & Credits
        # -------------------------------------------------------------
        footer = ctk.CTkFrame(self, height=28, fg_color="transparent")
        footer.pack(fill="x", padx=18, pady=(2, 10))

        self.status_lbl = ctk.CTkLabel(
            footer,
            text="Status: Ready",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.status_lbl.pack(side="left")

        credits_lbl = ctk.CTkLabel(
            footer,
            text="Candidate: Rupali Goyal • Lacritz Internship Assignment (Deadline: 04 Nov 2026)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray40", "gray65")
        )
        credits_lbl.pack(side="right")

    def _toggle_theme(self):
        """Switches between dark and light appearance modes."""
        current = ctk.get_appearance_mode()
        ctk.set_appearance_mode("Light" if current == "Dark" else "Dark")

    def _set_status(self, text: str):
        self.status_lbl.configure(text=f"Status: {text}")

    def _process_ui_queue(self):
        """Processes thread-safe UI updates on main thread."""
        try:
            while True:
                callback = self._ui_queue.get_nowait()
                try:
                    callback()
                except Exception as e:
                    print(f"[UI Queue Error] {e}")
                self._ui_queue.task_done()
        except queue.Empty:
            pass

        # If recording, update visual audio meter & timer
        if self.recorder.is_recording:
            dur = self.recorder.get_recording_duration()
            mins = int(dur // 60)
            secs = int(dur % 60)
            self.rec_status_lbl.configure(
                text=f"🔴 Recording: {mins:02d}:{secs:02d}",
                text_color="#ef4444"
            )
            # Smooth volume level bounce
            self.volume_bar.set(self.recorder.current_volume)
        else:
            self.volume_bar.set(0.0)

        # Schedule next loop at 50ms
        self.after(50, self._process_ui_queue)

    # -----------------------------------------------------------------
    # Audio Recording Handlers
    # -----------------------------------------------------------------
    def _on_start_recording(self):
        """Handles Start Recording button click."""
        # Determine device index
        device_str = self.mic_dropdown.get()
        dev_idx = None
        if device_str.startswith("[") and "]" in device_str:
            try:
                dev_idx = int(device_str[1:device_str.index("]")])
            except ValueError:
                dev_idx = None

        try:
            self.recorder.start_recording(device_index=dev_idx)
            self.btn_start.configure(state="disabled", fg_color="#6ee7b7")
            self.btn_stop.configure(state="normal", fg_color="#ef4444")
            self._set_status("Recording audio from microphone...")
        except AudioRecorderError as e:
            self._set_status(f"Microphone error: {e}")
            tk.messagebox.showwarning(
                "Microphone Notice",
                f"{e}\n\nTip: You can also use 'Load Sample Question' to test without an active mic!"
            )
        except Exception as e:
            self._set_status(f"Recording error: {e}")

    def _on_stop_recording(self):
        """Handles Stop Recording button click & triggers local STT."""
        self.btn_start.configure(state="normal", fg_color="#10b981")
        self.btn_stop.configure(state="disabled", fg_color="#71717a")
        self.rec_status_lbl.configure(text="Processing audio...", text_color="gray65")
        self._set_status("Transcribing speech with local faster-whisper...")

        # Run transcription in a background thread
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _transcribe_worker(self):
        """Background thread collecting audio buffer and transcribing."""
        try:
            audio_data = self.recorder.stop_recording()
            duration = len(audio_data) / 16000.0 if len(audio_data) > 0 else 0.0

            if duration < 0.4:
                self._ui_queue.put(lambda: self._set_status("Recording was too short to transcribe."))
                self._ui_queue.put(lambda: self.rec_status_lbl.configure(text="Idle • Ready", text_color="gray65"))
                return

            # Perform local speech-to-text
            text = self.stt.transcribe_audio(audio_data, sample_rate=16000)

            def update_ui():
                if text.strip():
                    self.txt_transcription.delete("1.0", "end")
                    self.txt_transcription.insert("1.0", text.strip())
                    self._set_status(f"Transcription complete ({duration:.1f}s audio)")
                else:
                    self._set_status("No speech detected in audio recording.")
                self.rec_status_lbl.configure(text="Idle • Ready", text_color="gray65")

            self._ui_queue.put(update_ui)

        except Exception as e:
            def on_err():
                self._set_status(f"Transcription error: {e}")
                self.rec_status_lbl.configure(text="Idle • Ready", text_color="gray65")
            self._ui_queue.put(on_err)

    def _load_next_sample_question(self):
        """Cycles through pre-configured technical questions for testing."""
        sample = SAMPLE_QUESTIONS[self._sample_index % len(SAMPLE_QUESTIONS)]
        self._sample_index += 1
        self.txt_transcription.delete("1.0", "end")
        self.txt_transcription.insert("1.0", sample["question"])
        self._set_status(f"Loaded sample: {sample['title']}")

    def _clear_transcription(self):
        """Clears transcription text area."""
        self.txt_transcription.delete("1.0", "end")
        self._set_status("Transcription cleared")

    # -----------------------------------------------------------------
    # AI Answer Progressive Generation
    # -----------------------------------------------------------------
    def _on_backend_changed(self, value: str):
        """Updates model options when backend is changed."""
        if "ollama" in value.lower():
            if not self.llm.ollama.is_available():
                self.llm_status_badge.configure(
                    text="○ Ollama Daemon Not Detected",
                    text_color="#ef4444"
                )
            else:
                self.llm_status_badge.configure(
                    text="● Ollama Daemon Active",
                    text_color="#10b981"
                )

    def _on_suggest_answer(self):
        """Handles 'Suggest Answer' button click to stream local AI response."""
        prompt = self.txt_transcription.get("1.0", "end").strip()
        if not prompt or prompt.startswith("Click 'Start Recording'"):
            tk.messagebox.showinfo(
                "No Question",
                "Please record audio or enter a question first before clicking 'Suggest Answer'."
            )
            return

        # Prepare progressive streaming display
        self.txt_answer.delete("1.0", "end")
        self.btn_suggest.configure(state="disabled", fg_color="#93c5fd")
        self.btn_stop_ai.configure(state="normal", fg_color="#ef4444")
        self.lbl_stats.configure(text="Generating response (streaming tokens)...", text_color="#3b82f6")
        self._set_status("AI is generating progressive answer...")

        self._stream_token_count = 0
        self._stream_start_time = time.time()

        selected_backend = self.backend_dropdown.get()
        selected_model = self.model_dropdown.get()

        # Call LLM engine
        self.llm.start_streaming(
            prompt=prompt,
            on_token=self._on_token_received,
            on_complete=self._on_generation_complete,
            on_error=self._on_generation_error,
            backend_name=selected_backend,
            model_name=selected_model
        )

    def _on_token_received(self, token: str):
        """Thread-safe handler for incoming progressive tokens."""
        def append():
            self._stream_token_count += 1
            self.txt_answer.insert("end", token)
            self.txt_answer.see("end")  # Auto-scroll to latest token

            # Update live stats
            elapsed = max(0.01, time.time() - self._stream_start_time)
            speed = self._stream_token_count / elapsed
            self.lbl_stats.configure(
                text=f"Streaming: {self._stream_token_count} chunks | {speed:.1f} tok/s | {elapsed:.1f}s",
                text_color="#10b981"
            )

        self._ui_queue.put(append)

    def _on_generation_complete(self):
        """Handler when AI generation finishes successfully."""
        def finish():
            self.btn_suggest.configure(state="normal", fg_color="#3b82f6")
            self.btn_stop_ai.configure(state="disabled", fg_color="#71717a")
            elapsed = max(0.01, time.time() - self._stream_start_time)
            speed = self._stream_token_count / elapsed
            self.lbl_stats.configure(
                text=f"✓ Complete ({self._stream_token_count} chunks in {elapsed:.1f}s • {speed:.1f} tok/s)",
                text_color="#10b981"
            )
            self._set_status("AI answer generation completed successfully.")

        self._ui_queue.put(finish)

    def _on_generation_error(self, err_msg: str):
        """Handler for generation errors."""
        def err():
            self.btn_suggest.configure(state="normal", fg_color="#3b82f6")
            self.btn_stop_ai.configure(state="disabled", fg_color="#71717a")
            self.lbl_stats.configure(text="⚠️ Generation interrupted / error", text_color="#ef4444")
            self.txt_answer.insert("end", f"\n\n[Error]: {err_msg}")
            self._set_status(f"Generation error: {err_msg}")

        self._ui_queue.put(err)

    def _on_stop_generation(self):
        """Cancels streaming generation."""
        self.llm.stop()
        self.btn_suggest.configure(state="normal", fg_color="#3b82f6")
        self.btn_stop_ai.configure(state="disabled", fg_color="#71717a")
        self.lbl_stats.configure(text="Generation stopped by user.", text_color="gray60")
        self._set_status("Generation stopped.")

    def _copy_answer_to_clipboard(self):
        """Copies the generated answer to system clipboard."""
        answer = self.txt_answer.get("1.0", "end").strip()
        if answer:
            self.clipboard_clear()
            self.clipboard_append(answer)
            self._set_status("Answer copied to clipboard! 📋")


def run_app():
    """Main launcher."""
    app = ParakeetDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
