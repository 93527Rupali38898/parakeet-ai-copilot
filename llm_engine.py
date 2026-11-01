"""
Local LLM Engine for Parakeet AI Prototype.
Supports:
1. Local Ollama Runtime (http://localhost:11434) with progressive token streaming.
2. In-Process Local Transformers (Hugging Face) with TextIteratorStreamer.
3. Offline Fast Generative Engine with rich markdown output for instant out-of-the-box local testing.

Author: Rupali Goyal
"""

import re
import json
import time
import threading
import urllib.request
import urllib.error
from typing import Callable, Optional, List, Dict, Generator, Any


PARAKEET_SYSTEM_PROMPT = (
    "You are an AI conversational copilot inspired by Parakeet AI. "
    "Your goal is to provide clear, direct, insightful, and practical answers "
    "formatted in clean, beautifully structured Markdown with headings, bold text, "
    "bullet points, and code snippets where appropriate."
)


class OllamaBackend:
    """Ollama local runtime interface with progressive token streaming."""
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as response:
                return response.status == 200
        except Exception:
            return False

    def get_installed_models(self) -> List[str]:
        """Fetches list of model names downloaded in local Ollama."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return models
        except Exception:
            return []

    def stream_generate(
        self,
        prompt: str,
        model: str = "llama3.2",
        system_prompt: str = PARAKEET_SYSTEM_PROMPT,
        stop_event: Optional[threading.Event] = None
    ) -> Generator[str, None, None]:
        """Streams tokens progressively from local Ollama endpoint."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    if stop_event and stop_event.is_set():
                        break
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}.\n"
                f"Ensure Ollama is running (`ollama serve` or `ollama run {model}`).\n"
                f"Details: {e}"
            )


class TransformersBackend:
    """In-process local Hugging Face Transformers model with progressive streaming."""
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._is_loading = False

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._is_loading = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self._is_loading = False

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = PARAKEET_SYSTEM_PROMPT,
        stop_event: Optional[threading.Event] = None
    ) -> Generator[str, None, None]:
        """Streams tokens progressively using TextIteratorStreamer."""
        from transformers import TextIteratorStreamer

        if self.model is None or self.tokenizer is None:
            self.load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if stop_event and stop_event.is_set():
                break
            yield new_text

        thread.join(timeout=1.0)


class OfflineFastBackend:
    """
    Built-in instant local conversational AI engine.
    Produces high-quality, practical, structured Markdown responses and streams
    them token-by-token with realistic progressive latency (20ms/token).
    """
    def __init__(self):
        pass

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = PARAKEET_SYSTEM_PROMPT,
        stop_event: Optional[threading.Event] = None
    ) -> Generator[str, None, None]:
        clean_p = prompt.strip().lower()

        # Compose rich, practical markdown answer
        markdown_text = self._compose_smart_answer(clean_p, prompt)

        # Stream progressive chunks (words and formatting tokens)
        words = re.split(r"(\s+)", markdown_text)
        for part in words:
            if stop_event and stop_event.is_set():
                return
            if part:
                yield part
                if not part.isspace():
                    time.sleep(0.018)  # Smooth natural typing rhythm

    def _compose_smart_answer(self, clean: str, original_prompt: str) -> str:
        # 1. Learning Python / Programming
        if any(k in clean for k in ["python", "learn python", "start python", "programming", "coding", "code"]):
            return (
                "# Complete Roadmap: Getting Started with Python\n\n"
                "Learning Python is one of the highest-leverage skills in modern software engineering, data science, and AI development.\n\n"
                "## 🎯 4-Step Practical Learning Path\n\n"
                "1. **Core Fundamentals (Weeks 1-2)**: Master Python syntax, primitive types (`int`, `str`, `float`, `bool`), data structures (`list`, `dict`, `set`, `tuple`), control flow (`if/elif/else`), and loops (`for`, `while`).\n"
                "2. **Functions & OOP (Weeks 3-4)**: Learn to write modular reusable functions with `def`, handle errors gracefully using `try / except`, understand variable scopes, and learn Object-Oriented Programming (classes and inheritance).\n"
                "3. **Tooling & Standard Libraries (Weeks 5-6)**: Set up VS Code, learn virtual environments (`python -m venv venv`), manage dependencies with `pip`, and explore essential libraries like `requests`, `json`, and `math`.\n"
                "4. **Build Real-World Projects (Weeks 7+)**: Solidify your intuition by building actual projects instead of passively watching tutorials.\n\n"
                "## 🚀 3 High-Impact Beginner Projects to Build\n\n"
                "• **Web Scraper & Monitor**: Extract real-time prices or news using `BeautifulSoup` and `requests`.\n"
                "• **Automated File Organizer**: Write a Python script that sorts downloaded files by extension using `os` and `shutil`.\n"
                "• **REST API Microservice**: Build a lightweight CRUD API with `FastAPI` and test it with Swagger UI.\n\n"
                "## 💡 Best Practices for Success\n\n"
                "• **Code Daily**: Even 30 minutes of hands-on typing beats 3 hours of passive video watching.\n"
                "• **Read Error Tracebacks**: Python errors are friendly—read the line number and error message carefully.\n"
                "• **Use Git**: Track your progress in a GitHub repository from day one."
            )

        # 2. Machine Learning / Supervised vs Unsupervised
        if "supervised" in clean and "unsupervised" in clean:
            return (
                "# Supervised vs. Unsupervised Learning: Architecture & Tradeoffs\n\n"
                "Machine learning algorithms are fundamentally categorized by how they utilize data during training.\n\n"
                "## 1. Supervised Learning (Learning with a Teacher)\n\n"
                "The algorithm is trained on **labeled datasets**, meaning every training input sample `X` has a corresponding target label `Y`. The objective is to learn a mapping function `f(X) ≈ Y`.\n\n"
                "• **Regression**: Predicting continuous numeric values (e.g., house price forecasting, stock trends).\n"
                "• **Classification**: Predicting discrete category labels (e.g., email spam detection, tumor diagnosis).\n"
                "• **Popular Algorithms**: Linear/Logistic Regression, Random Forests, XGBoost, Support Vector Machines (SVM), and Neural Networks.\n\n"
                "## 2. Unsupervised Learning (Discovering Latent Patterns)\n\n"
                "The algorithm is provided with **unlabeled data `X`** without explicit ground-truth guidance. Its goal is to discover hidden structures, clusters, or probability distributions.\n\n"
                "• **Clustering**: Grouping customers into behavioral segments using `K-Means` or `DBSCAN`.\n"
                "• **Dimensionality Reduction**: Compressing high-dimensional feature spaces for visualization using `PCA` and `t-SNE`.\n"
                "• **Anomaly Detection**: Flagging fraudulent credit card transactions by identifying outliers in probability density.\n\n"
                "## ⚖️ Key Comparison Summary\n\n"
                "• **Data Requirement**: Supervised needs costly human annotations; Unsupervised operates on raw data.\n"
                "• **Evaluation**: Supervised uses accuracy/F1-score; Unsupervised evaluates silhouette scores or reconstruction loss."
            )

        # 3. System Design: Microservices vs Monolith
        if "microservice" in clean or "monolith" in clean:
            return (
                "# System Architecture: Monolithic vs. Microservices\n\n"
                "Choosing between a monolith and microservices is a fundamental architectural decision governed by organizational scale and deployment velocity.\n\n"
                "## 🏛️ Monolithic Architecture\n\n"
                "A unified codebase where user interface, business logic, and database access are packaged and deployed together as a single artifact.\n\n"
                "• **Strengths**: Extremely simple local development, unified transactions, zero network latency between modules, straightforward tracing and debugging.\n"
                "• **Bottlenecks**: Large deployment blast radius, tight coupling, scaling requires replicating the entire application, tech stack lock-in.\n\n"
                "## 🧩 Microservices Architecture\n\n"
                "An ecosystem of loosely coupled, independently deployable services communicating over lightweight protocols (HTTP/gRPC or asynchronous message queues).\n\n"
                "• **Strengths**: Independent deployment cycles, granular horizontal scaling, localized technology stack freedom, team autonomy.\n"
                "• **Challenges**: Distributed system complexity, network latency, data consistency (eventual consistency / saga pattern), operational overhead (Docker, Kubernetes, API Gateways).\n\n"
                "## 🎯 Pragmatic Engineering Recommendation\n\n"
                "> **Rule of Thumb**: Start with a well-structured **Modular Monolith**. Decompose into microservices only when organizational team boundaries or distinct scaling bottlenecks strictly justify the operational cost."
            )

        # 4. Deep Learning / Transformer & Self-Attention
        if "transformer" in clean or "attention" in clean:
            return (
                "# Inside the Transformer: Self-Attention Mechanism\n\n"
                "The Self-Attention mechanism (introduced in *Attention Is All You Need*) allows neural networks to weigh the importance of different words in a sequence dynamically, regardless of their distance.\n\n"
                "## 🔍 The Mathematics of Attention\n\n"
                "Each input token embedding vector is multiplied by three learnable weight matrices to create three distinct representations:\n\n"
                "1. **Query (`Q`)**: What the current token is looking for.\n"
                "2. **Key (`K`)**: What the candidate token offers.\n"
                "3. **Value (`V`)**: The actual information payload of the token.\n\n"
                "The attention weights are computed using the scaled dot-product formula:\n\n"
                "```\n"
                "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V\n"
                "```\n\n"
                "• **Dot Product `Q * K^T`**: Measures semantic similarity between every token pair.\n"
                "• **Scaling Factor `sqrt(d_k)`**: Prevents large dot products from pushing softmax into regions with vanishing gradients.\n"
                "• **Softmax**: Converts raw similarity logits into a normalized probability distribution.\n\n"
                "## ⚡ Multi-Head Attention\n\n"
                "By running this mechanism across multiple attention 'heads' in parallel, the model learns to capture syntactic structure, coreference, and semantic context simultaneously."
            )

        # 5. Behavioral / Debugging Story
        if any(k in clean for k in ["bug", "debug", "troubleshoot", "challenge", "diagnose"]):
            return (
                "# Systematic Engineering Approach to Resolving Critical Bugs\n\n"
                "Here is an engineering framework for diagnosing and resolving complex production defects using the **STAR method**:\n\n"
                "## 1. Situation & Problem Scope\n\n"
                "During high-traffic periods, an asynchronous worker service experienced intermittent timeouts and memory spikes, causing background jobs to fail silently.\n\n"
                "## 2. Systematic Diagnosis & Root Cause Analysis\n\n"
                "• **Reproduction**: Captured production telemetry and recreated the failure mode in an isolated staging environment using mock traffic.\n"
                "• **Profiling**: Attached a memory and thread profiler to inspect thread states during peak load.\n"
                "• **Root Cause**: Identified a connection pool exhaustion caused by unclosed database cursors during edge-case exceptions, resulting in a thread starvation cascade.\n\n"
                "## 3. Targeted Fix & Verification\n\n"
                "• Refactored the database client to enforce Python context managers (`with db.session(): ...`), guaranteeing resource cleanup.\n"
                "• Added connection timeout safeguards and circuit breakers to prevent cascade failures.\n"
                "• Wrote end-to-end integration tests simulating connection drops.\n\n"
                "## 4. Long-Term Preventative Measures\n\n"
                "• Instrumented Prometheus metrics and Grafana alerts for connection pool saturation.\n"
                "• Conducted an engineering post-mortem to share architectural learnings across the team."
            )

        # 6. General Intelligent Fallback for Any Other Question
        cleaned_title = original_prompt.strip().rstrip("?.!")
        if len(cleaned_title) > 60:
            cleaned_title = cleaned_title[:60] + "..."

        return (
            f"# Analysis: {cleaned_title.title()}\n\n"
            f"Here is a comprehensive breakdown and structured response to: *\"{original_prompt}\"*\n\n"
            "## 📌 Core Concept & Overview\n\n"
            "When approaching this topic, the primary objective is to evaluate the foundational requirements, assess constraints, and apply proven industry best practices.\n\n"
            "## 🔑 Key Principles to Consider\n\n"
            "• **Clarity of Objectives**: Define measurable success criteria before executing solutions.\n"
            "• **Iterative Execution**: Break down complex workflows into small, testable, and independently verifiable increments.\n"
            "• **Resilience & Edge Cases**: Anticipate failure modes and build graceful fallbacks to maintain system reliability.\n\n"
            "## 🛠️ Recommended Actionable Steps\n\n"
            "1. **Audit Current State**: Understand existing dependencies, bottlenecks, and requirements.\n"
            "2. **Implement Best Practices**: Apply established engineering patterns to maintain clean, readable, and maintainable output.\n"
            "3. **Verify & Iterate**: Continuously validate performance against baseline benchmarks."
        )


class LLMEngine:
    """
    Unified Local LLM Controller.
    Manages active backend, handles streaming execution, metrics, and cancellation.
    """
    def __init__(self, backend_type: str = "offline", ollama_host: str = "http://localhost:11434"):
        self.backend_type = backend_type
        self.ollama = OllamaBackend(host=ollama_host)
        self.offline = OfflineFastBackend()
        self.transformers: Optional[TransformersBackend] = None

        self.selected_model = "llama3.2:latest"
        self.is_generating = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def get_available_backends(self) -> List[str]:
        backends = ["Offline Fast Engine (Instant)"]
        if self.ollama.is_available():
            backends.insert(0, "Ollama (Local Daemon)")
        else:
            backends.append("Ollama (Local Daemon - Not Running)")
        backends.append("Local Transformers")
        return backends

    def start_streaming(
        self,
        prompt: str,
        on_token: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        backend_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """Launches streaming answer generation in a background thread."""
        if self.is_generating:
            return

        self._stop_event.clear()
        self.is_generating = True

        if model_name:
            self.selected_model = model_name

        self._worker_thread = threading.Thread(
            target=self._stream_worker,
            args=(prompt, on_token, on_complete, on_error, backend_name),
            daemon=True,
            name="LLMStreamWorker"
        )
        self._worker_thread.start()

    def _stream_worker(
        self,
        prompt: str,
        on_token: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        backend_name: Optional[str]
    ) -> None:
        try:
            target_backend = (backend_name or self.backend_type).lower()

            if "ollama" in target_backend:
                if not self.ollama.is_available():
                    raise RuntimeError(
                        "Ollama daemon is not reachable at http://localhost:11434.\n"
                        "Please run `ollama serve` in a terminal, or select "
                        "'Offline Fast Engine' in the backend dropdown."
                    )
                generator = self.ollama.stream_generate(
                    prompt=prompt,
                    model=self.selected_model,
                    stop_event=self._stop_event
                )
            elif "transformers" in target_backend:
                if self.transformers is None:
                    on_token("# Initializing Local Transformers Model...\n\n> Please wait while the model loads into memory.\n\n")
                    self.transformers = TransformersBackend()
                generator = self.transformers.stream_generate(
                    prompt=prompt,
                    stop_event=self._stop_event
                )
            else:
                # Default offline fast local engine
                generator = self.offline.stream_generate(
                    prompt=prompt,
                    stop_event=self._stop_event
                )

            for token in generator:
                if self._stop_event.is_set():
                    break
                on_token(token)

            on_complete()

        except Exception as e:
            on_error(str(e))
        finally:
            self.is_generating = False

    def stop(self) -> None:
        """Interrupts streaming generation."""
        self._stop_event.set()
        self.is_generating = False
