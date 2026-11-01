# Parakeet AI — Desktop Prototype

> **Candidate Name:** Rupali Goyal  
> **Submission Deadline:** 04 November, 2026  
> **Target Platforms:** Windows & macOS  
> **Core Technologies:** Python, CustomTkinter, Faster-Whisper, Ollama / Hugging Face Local Transformers

---

## Overview

This project is a desktop application prototype inspired by **Parakeet AI**. It provides an intelligent, real-time desktop copilot that:

1. **Captures microphone audio** with clean, responsive **Start** and **Stop** buttons.
2. **Transcribes spoken speech locally** using a quantized, high-efficiency Whisper speech-to-text model (`faster-whisper-tiny` with `int8` quantization running on CPU/GPU).
3. **Generates contextual answers** when the user clicks **“Suggest Answer”**.
4. **Displays the AI response progressively** (streaming token-by-token in real time) with live metrics (tokens streamed, generation speed, elapsed latency).
5. **Runs 100% locally** without external paid cloud APIs, supporting **Ollama** runtimes (e.g., `llama3.2`, `mistral`, `phi3`), **Local Transformers**, and an instant **Offline Fast Engine** for zero-setup evaluation.

---

## Architecture & Workflow

```
       +---------------------------------------------+
       |               Microphone Input              |
       +---------------------------------------------+
                              |
                     [ Start Recording ]
                              |
              +---------------+---------------+
              |   audio_recorder.py (Buffer)  |
              |   - Live Volume RMS Meter     |
              |   - 16kHz Float32 Capture     |
              +---------------+---------------+
                              |
                     [ Stop Recording ]
                              |
              +---------------+---------------+
              |        stt_engine.py          |
              |   - Local Faster-Whisper      |
              |   - VAD Voice Activity Filter |
              +---------------+---------------+
                              |
            Transcribed Question in Text Area
                              |
                    [  Suggest Answer ]
                              |
              +---------------+---------------+
              |        llm_engine.py          |
              |   - Local Ollama Runtime      |
              |   - In-Process Transformers   |
              |   - Progressive Token Stream  |
              +---------------+---------------+
                              |
           Progressive Token-by-Token Streaming
           UI Autoscroll + Typing Cursor Effect
```

---

## Key Features

* **Parakeet-Inspired Modern UI**: Designed with `customtkinter` featuring Dark/Light theme toggling, rounded visual cards, and responsive layout.
* **Live Audio Visualizer**: Real-time volume level meter reflecting microphone activity during recording.
* **Editable Transcription**: Users can review or fine-tune the transcribed question before submitting it to the AI.
* **Zero-Mic Sample Testing**: Built-in `Load Sample Question` button allows testing technical, system design, and behavioral questions without needing a physical microphone or speaking aloud.
* **True Progressive Streaming**: Answers stream incrementally chunk-by-chunk using a thread-safe message queue, ensuring zero UI freezing.
* **Local STT & LLM**: No cloud subscription or internet connectivity required during inference.

---

## Prerequisites

* **Python:** Python 3.10, 3.11, or 3.12 installed.
* **Microphone:** Standard built-in microphone, USB headset, or Bluetooth hands-free audio.
* *(Optional)* **Ollama:** If you wish to use Ollama for local LLM inference ([Download Ollama](https://ollama.com)).

---

## Installation

### 1. Clone the repository & navigate to project folder
```bash
git clone <repo-url>
cd Lacritz
```

### 2. Create and activate a virtual environment (recommended)
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Option A: One-Click Launchers
* **Windows:** Double-click `run.bat`
* **macOS / Linux:** Execute `./run.sh`

### Option B: Command Line
```bash
python main.py
```

#### Optional CLI Flags
```bash
python main.py --theme dark --backend offline
python main.py --backend ollama --model llama3.2
```

---

## Using Ollama for Local LLM (Optional)

If you have Ollama installed, the app can automatically detect and stream from your local Ollama models:

1. **Start Ollama service:**
   ```bash
   ollama serve
   ```
2. **Pull a fast, lightweight local model (e.g., Llama 3.2):**
   ```bash
   ollama run llama3.2
   ```
   *(or `ollama pull llama3.2:1b` for ultra-fast CPU inference)*
3. In the application's **LLM** dropdown, select **Ollama (Local Daemon)**.
4. The prototype will automatically connect to `http://localhost:11434` and stream tokens progressively!

> **Note:** If Ollama is not installed or not running, the application seamlessly defaults to the built-in **Offline Fast Engine**, allowing full feature testing with realistic progressive streaming!

---

## Automated Testing

A comprehensive integration test suite is included:

```bash
python test_pipeline.py
```

This verifies:
* Audio device enumeration and recorder initialization.
* Faster-Whisper local speech-to-text inference.
* Progressive token streaming from the LLM controller.
* Thread safety and callback delivery.

---

## Project Structure

```
Lacritz/
├── audio_recorder.py   # Microphone capture, volume level calculation, and buffers
├── stt_engine.py       # Local Faster-Whisper speech-to-text service & sample prompts
├── llm_engine.py       # Local LLM backends (Ollama, Transformers, Offline Fast Engine)
├── gui.py              # CustomTkinter GUI layout, progressive streaming, and events
├── main.py             # Application launcher and argument parser
├── test_pipeline.py    # Unit & integration verification test suite
├── requirements.txt    # Project dependencies
├── run.bat             # Windows one-click launcher
├── run.sh              # macOS/Linux launcher
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation
```

---

## Author

* **Author:** Rupali Goyal

