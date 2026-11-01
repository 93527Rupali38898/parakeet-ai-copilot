"""
Desktop GUI for Parakeet AI Prototype.
Built with CustomTkinter for a modern, sleek interface inspired by Parakeet AI.
Features rich real-time Markdown rendering, Start/Stop microphone recording,
live audio level visualizer, Whisper STT, and progressive streaming LLM answer generation.

Author: Rupali Goyal
"""

import time
import queue
import threading
import tkinter as tk
from typing import Optional, List, Dict

import customtkinter as ctk

from audio_recorder import AudioRecorder, AudioRecorderError
from stt_engine import STTEngine
from llm_engine import LLMEngine
from markdown_viewer import MarkdownViewer

# Modern Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class ParakeetDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Parakeet AI — Desktop Copilot Prototype")
        self.geometry("960x840")
        self.minsize(820, 700)

        # Core Engines
        self.recorder = AudioRecorder(sample_rate=16000)
        self.stt = STTEngine(model_size="tiny", device="cpu", compute_type="int8")
        self.llm = LLMEngine(backend_type="offline")

        # Application state
        self._poll_timer_id = None
        self._stream_token_count = 0
        self._stream_start_time = 0.0
        self._ui_queue = queue.Queue()
        self._view_raw_markdown = False

        # Build UI layout
        self._setup_ui()

        # Start UI event polling loop
        self._process_ui_queue()

        # Preload STT model in background
        threading.Thread(target=self._preload_stt, daemon=True).start()

        # Populate audio devices & Ollama status in background
        threading.Thread(target=self._detect_system_devices, daemon=True).start()

    def _preload_stt(self):
        """Preloads faster-whisper model into memory silently."""
        try:
            self.stt.load_model()
            self._ui_queue.put(lambda: self._set_status("Speech engine ready (faster-whisper tiny • int8)"))
        except Exception as e:
            self._ui_queue.put(lambda: self._set_status(f"Speech engine initialization warning: {e}"))

    def _detect_system_devices(self):
        """Scans for available microphones and local Ollama server."""
        devices = self.recorder.get_input_devices()
        ollama_running = self.llm.ollama.is_available()
        ollama_models = self.llm.ollama.get_installed_models() if ollama_running else []

        def update():
            # Update Mic selector
            device_names = [f"[{idx}] {name[:32]}" for idx, name in devices]
            if not device_names:
                device_names = ["Default System Microphone"]
            self.mic_dropdown.configure(values=device_names)
            self.mic_dropdown.set(device_names[0])

            # Update LLM Backend
            if ollama_running:
                self.backend_dropdown.set("Ollama (Local Daemon)")
                self.llm_status_badge.configure(
                    text="● Ollama Active",
                    text_color="#10b981"
                )
                if ollama_models:
                    self.model_dropdown.configure(values=ollama_models)
                    self.model_dropdown.set(ollama_models[0])
            else:
                self.backend_dropdown.set("Offline Fast Engine (Instant)")
                self.llm_status_badge.configure(
                    text="○ Local Fast Engine",
                    text_color="#38bdf8"
                )

        self._ui_queue.put(update)

    def _setup_ui(self):
        """Constructs the sleek Parakeet AI desktop interface."""
        # Top Header Frame
        header = ctk.CTkFrame(self, corner_radius=12, fg_color=("#e2e8f0", "#0f172a"))
        header.pack(fill="x", padx=16, pady=(12, 6))

        # Title & Subtitle Box
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=10)

        brand_row = ctk.CTkFrame(title_box, fg_color="transparent")
        brand_row.pack(anchor="w")

        title_lbl = ctk.CTkLabel(
            brand_row,
            text="🦜 PARAKEET AI",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_lbl.pack(side="left")

        tag_badge = ctk.CTkLabel(
            brand_row,
            text="STUDIO DESKTOP",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#064e3b",
            text_color="#34d399",
            corner_radius=6,
            padx=8,
            pady=2
        )
        tag_badge.pack(side="left", padx=10)

        subtitle_lbl = ctk.CTkLabel(
            title_box,
            text="Local Speech-to-Text • Progressive LLM Streaming • Intelligent Answer Copilot",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray65")
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Top Right Controls (Theme & System Badges)
        right_header = ctk.CTkFrame(header, fg_color="transparent")
        right_header.pack(side="right", padx=16, pady=10)

        self.llm_status_badge = ctk.CTkLabel(
            right_header,
            text="Detecting local engines...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray60"
        )
        self.llm_status_badge.pack(side="top", anchor="e", pady=(0, 4))

        theme_switch = ctk.CTkSwitch(
            right_header,
            text="Dark Theme",
            command=self._toggle_theme,
            font=ctk.CTkFont(size=11)
        )
        theme_switch.select()
        theme_switch.pack(side="top", anchor="e")

        # Main Layout Frame
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=16, pady=4)

        # -------------------------------------------------------------
        # CARD 1: Audio Capture & Transcription Section
        # -------------------------------------------------------------
        audio_card = ctk.CTkFrame(main_frame, corner_radius=12, fg_color=("#ffffff", "#1e293b"))
        audio_card.pack(fill="x", pady=5)

        # Card Title Bar with Device Selector
        bar1 = ctk.CTkFrame(audio_card, fg_color="transparent")
        bar1.pack(fill="x", padx=14, pady=(10, 4))

        sec1_title = ctk.CTkLabel(
            bar1,
            text="🎙️ Microphone Voice Input",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        sec1_title.pack(side="left")

        # Device selector dropdown
        self.mic_dropdown = ctk.CTkOptionMenu(
            bar1,
            values=["Default Microphone"],
            width=220,
            height=28,
            font=ctk.CTkFont(size=11)
        )
        self.mic_dropdown.pack(side="right")

        device_lbl = ctk.CTkLabel(bar1, text="Mic Device:", font=ctk.CTkFont(size=11), text_color="gray60")
        device_lbl.pack(side="right", padx=(0, 6))

        # Recording Control Bar (Start, Stop, Live Waveform / Meter)
        btn_bar = ctk.CTkFrame(audio_card, fg_color="transparent")
        btn_bar.pack(fill="x", padx=14, pady=4)

        self.btn_start = ctk.CTkButton(
            btn_bar,
            text="▶  Start Recording",
            command=self._on_start_recording,
            fg_color="#10b981",
            hover_color="#059669",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=165,
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
            width=165,
            height=36
        )
        self.btn_stop.pack(side="left", padx=(0, 14))

        # Recording timer and live visualizer
        self.rec_status_lbl = ctk.CTkLabel(
            btn_bar,
            text="Idle • Ready",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray65"
        )
        self.rec_status_lbl.pack(side="left", padx=(0, 12))

        self.volume_bar = ctk.CTkProgressBar(btn_bar, width=150, height=12)
        self.volume_bar.set(0.0)
        self.volume_bar.pack(side="left", padx=(0, 10))

        # Quick Example Chips Row
        chip_bar = ctk.CTkFrame(audio_card, fg_color="transparent")
        chip_bar.pack(fill="x", padx=14, pady=(6, 2))

        lbl_trans = ctk.CTkLabel(
            chip_bar,
            text="Transcribed Question / Prompt:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_trans.pack(side="left")

        # Quick preset buttons for instant testing
        presets = [
            ("🐍 Learn Python", "I want to start learning Python. What is the recommended roadmap and best beginner projects?"),
            ("🤖 ML Concepts", "Can you explain the difference between supervised and unsupervised learning with practical examples?"),
            ("🏗️ System Design", "What are microservices, and what are the main tradeoffs compared to a monolithic architecture?"),
            ("🔍 Attention", "How does the self-attention mechanism work in a Transformer model?"),
            ("🐞 Debug Bug", "Tell me about a challenging technical bug you encountered and how you systematically diagnosed it.")
        ]

        btn_clear = ctk.CTkButton(
            chip_bar,
            text="🗑️ Clear",
            command=self._clear_transcription,
            width=65,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90")
        )
        btn_clear.pack(side="right")

        for label, text in reversed(presets):
            b = ctk.CTkButton(
                chip_bar,
                text=label,
                command=lambda t=text: self._set_prompt_text(t),
                width=100,
                height=24,
                font=ctk.CTkFont(size=10),
                fg_color=("gray85", "gray25"),
                text_color=("gray10", "gray90")
            )
            b.pack(side="right", padx=3)

        # Text area for transcription
        self.txt_transcription = ctk.CTkTextbox(
            audio_card,
            height=68,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            wrap="word"
        )
        self.txt_transcription.pack(fill="x", padx=14, pady=(2, 10))
        self.txt_transcription.insert(
            "1.0",
            "I want to start learning Python. What is the recommended roadmap and best beginner projects?"
        )

        # -------------------------------------------------------------
        # CARD 2: AI Answer Suggestion Section (Markdown Rendered)
        # -------------------------------------------------------------
        ai_card = ctk.CTkFrame(main_frame, corner_radius=12, fg_color=("#ffffff", "#1e293b"))
        ai_card.pack(fill="both", expand=True, pady=5)

        # AI Action & Backend Selector Bar
        ai_controls = ctk.CTkFrame(ai_card, fg_color="transparent")
        ai_controls.pack(fill="x", padx=14, pady=(10, 4))

        self.btn_suggest = ctk.CTkButton(
            ai_controls,
            text="✨  Suggest Answer",
            command=self._on_suggest_answer,
            fg_color="#4f46e5",
            hover_color="#4338ca",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=180,
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

        # Model selector dropdown
        self.model_dropdown = ctk.CTkOptionMenu(
            ai_controls,
            values=["llama3.2", "llama3.2:1b", "mistral", "phi3"],
            width=120,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        self.model_dropdown.pack(side="right", padx=(0, 10))

        # Answer Title & Live Stats Bar
        answer_bar = ctk.CTkFrame(ai_card, fg_color="transparent")
        answer_bar.pack(fill="x", padx=14, pady=(4, 2))

        lbl_answer = ctk.CTkLabel(
            answer_bar,
            text="AI Suggested Answer (Progressive Markdown):",
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

        self.btn_toggle_view = ctk.CTkButton(
            answer_bar,
            text="📖 Formatted View",
            command=self._toggle_view_mode,
            width=120,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90")
        )
        self.btn_toggle_view.pack(side="right", padx=6)

        # Rich Markdown Answer Display Widget
        # Underlying tk.Text wrapped inside CTkFrame with native styling
        text_container = ctk.CTkFrame(ai_card, corner_radius=8, fg_color=("#f8fafc", "#0f172a"))
        text_container.pack(fill="both", expand=True, padx=14, pady=(2, 10))

        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side="right", fill="y")

        self.native_text = tk.Text(
            text_container,
            wrap="word",
            font=("Segoe UI", 12),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
            yscrollcommand=scrollbar.set
        )
        self.native_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.native_text.yview)

        # Attach Markdown viewer
        self.markdown_viewer = MarkdownViewer(self.native_text, is_dark=True)
        initial_markdown = (
            "# Welcome to Parakeet AI Copilot\n\n"
            "This prototype runs local Speech-to-Text and streaming local LLM generation.\n\n"
            "## 🎙️ How to Use\n"
            "1. **Record Audio**: Click `▶ Start Recording` and speak into your microphone, or choose a quick prompt above.\n"
            "2. **Transcribe**: Click `⏹ Stop Recording` to transcribe audio with local Faster-Whisper.\n"
            "3. **Suggest Answer**: Click `✨ Suggest Answer` to stream the AI response progressively with formatted Markdown!"
        )
        self.markdown_viewer.render(initial_markdown, streaming=False)

        # -------------------------------------------------------------
        # Bottom Status Bar & Credits
        # -------------------------------------------------------------
        footer = ctk.CTkFrame(self, height=28, fg_color="transparent")
        footer.pack(fill="x", padx=18, pady=(2, 8))

        self.status_lbl = ctk.CTkLabel(
            footer,
            text="Status: Ready",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.status_lbl.pack(side="left")

        credits_lbl = ctk.CTkLabel(
            footer,
            text="Author: Rupali Goyal • Lacritz Internship Assignment (Deadline: 04 Nov 2026)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray40", "gray65")
        )
        credits_lbl.pack(side="right")

    def _toggle_theme(self):
        """Switches between dark and light appearance modes."""
        current = ctk.get_appearance_mode()
        is_dark = (current != "Dark")
        ctk.set_appearance_mode("Dark" if is_dark else "Light")
        self.markdown_viewer.set_theme(is_dark=is_dark)

    def _toggle_view_mode(self):
        """Toggles between formatted visual Markdown and raw Markdown text."""
        self._view_raw_markdown = not self._view_raw_markdown
        if self._view_raw_markdown:
            self.btn_toggle_view.configure(text="📄 Raw View")
            # Show raw text
            self.native_text.configure(state="normal")
            self.native_text.delete("1.0", "end")
            self.native_text.insert("1.0", self.markdown_viewer.raw_text)
            self.native_text.configure(state="disabled")
        else:
            self.btn_toggle_view.configure(text="📖 Formatted View")
            self.markdown_viewer.render(self.markdown_viewer.raw_text, streaming=False)

    def _set_prompt_text(self, text: str):
        """Sets the transcription text area with preset prompt."""
        self.txt_transcription.delete("1.0", "end")
        self.txt_transcription.insert("1.0", text)
        self._set_status("Loaded preset question")

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
            self.volume_bar.set(self.recorder.current_volume)
        else:
            self.volume_bar.set(0.0)

        self.after(50, self._process_ui_queue)

    # -----------------------------------------------------------------
    # Audio Recording Handlers
    # -----------------------------------------------------------------
    def _on_start_recording(self):
        """Handles Start Recording button click."""
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
            self._set_status(f"Microphone notice: {e}")
            tk.messagebox.showwarning(
                "Microphone Notice",
                f"{e}\n\nTip: You can also use the preset chips above to test without an active microphone!"
            )
        except Exception as e:
            self._set_status(f"Recording error: {e}")

    def _on_stop_recording(self):
        """Handles Stop Recording button click & triggers local STT."""
        self.btn_start.configure(state="normal", fg_color="#10b981")
        self.btn_stop.configure(state="disabled", fg_color="#71717a")
        self.rec_status_lbl.configure(text="Processing audio...", text_color="gray65")
        self._set_status("Transcribing speech with local faster-whisper...")

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

    def _clear_transcription(self):
        """Clears transcription text area."""
        self.txt_transcription.delete("1.0", "end")
        self._set_status("Transcription cleared")

    # -----------------------------------------------------------------
    # AI Answer Progressive Generation
    # -----------------------------------------------------------------
    def _on_backend_changed(self, value: str):
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
        if not prompt:
            tk.messagebox.showinfo(
                "No Question",
                "Please record audio or select a question first before clicking 'Suggest Answer'."
            )
            return

        # Prepare progressive streaming display
        self.markdown_viewer.clear()
        self.btn_suggest.configure(state="disabled", fg_color="#818cf8")
        self.btn_stop_ai.configure(state="normal", fg_color="#ef4444")
        self.lbl_stats.configure(text="Generating response (streaming tokens)...", text_color="#38bdf8")
        self._set_status("AI is streaming progressive answer in rich Markdown...")

        self._stream_token_count = 0
        self._stream_start_time = time.time()

        selected_backend = self.backend_dropdown.get()
        selected_model = self.model_dropdown.get()

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
            self.markdown_viewer.append_chunk(token)

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
            self.markdown_viewer.finish_streaming()
            self.btn_suggest.configure(state="normal", fg_color="#4f46e5")
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
            self.markdown_viewer.finish_streaming()
            self.btn_suggest.configure(state="normal", fg_color="#4f46e5")
            self.btn_stop_ai.configure(state="disabled", fg_color="#71717a")
            self.lbl_stats.configure(text="⚠️ Generation interrupted / error", text_color="#ef4444")
            self.markdown_viewer.append_chunk(f"\n\n> **[Error]**: {err_msg}\n")
            self._set_status(f"Generation error: {err_msg}")

        self._ui_queue.put(err)

    def _on_stop_generation(self):
        """Cancels streaming generation."""
        self.llm.stop()
        self.markdown_viewer.finish_streaming()
        self.btn_suggest.configure(state="normal", fg_color="#4f46e5")
        self.btn_stop_ai.configure(state="disabled", fg_color="#71717a")
        self.lbl_stats.configure(text="Generation stopped by user.", text_color="gray60")
        self._set_status("Generation stopped.")

    def _copy_answer_to_clipboard(self):
        """Copies the generated answer to system clipboard."""
        answer = self.markdown_viewer.raw_text.strip()
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
