"""
Parakeet AI Desktop Prototype - Entry Point.
Launches the desktop GUI for local speech transcription and progressive AI answer generation.

Author: Rupali Goyal
Submission Date: November 2026
"""

import sys
import argparse

# Ensure standard UTF-8 console encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from gui import ParakeetDesktopApp, run_app


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parakeet AI - Local Speech-to-Text & Progressive AI Answer Desktop Prototype"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="offline",
        choices=["offline", "ollama", "transformers"],
        help="Default local LLM backend to use (default: offline)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Model name for Ollama backend (default: llama3.2)"
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="Initial UI appearance mode (default: dark)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print("=" * 65)
    print("🦜 Parakeet AI — Desktop Prototype")
    print("Candidate: Rupali Goyal")
    print(f"Default Backend: {args.backend} | Theme: {args.theme}")
    print("=" * 65)

    app = ParakeetDesktopApp()
    if args.theme.lower() == "light":
        import customtkinter as ctk
        ctk.set_appearance_mode("Light")

    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
