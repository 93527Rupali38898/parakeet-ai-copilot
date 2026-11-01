"""
Markdown Streaming Viewer for Tkinter / CustomTkinter.
Renders markdown elements (headings, bold, italics, bullets, numbered lists,
inline code, and code blocks) dynamically in real-time as tokens stream in.

Author: Rupali Goyal
"""

import re
import tkinter as tk
from typing import Optional


class MarkdownViewer:
    """
    Renders styled markdown into a Tkinter Text widget in real time.
    Handles headings (#, ##, ###), bold (**text**), italics (*text*),
    bullet points (- , * , •), numbered lists (1. , 2.), inline code (`code`),
    and multi-line code blocks (```...```).
    """
    def __init__(self, text_widget: tk.Text, is_dark: bool = True):
        self.widget = text_widget
        self.is_dark = is_dark
        self.raw_text = ""
        self.is_streaming = False
        self._cursor_visible = True
        self.setup_styles()

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self.setup_styles()
        self.render(self.raw_text, streaming=self.is_streaming)

    def setup_styles(self):
        """Configures typography, colors, and margins for markdown tags."""
        if self.is_dark:
            fg_main = "#f1f5f9"        # Slate 100
            fg_secondary = "#cbd5e1"   # Slate 300
            fg_bold = "#ffffff"        # Pure white
            fg_h1 = "#34d399"          # Emerald 400
            fg_h2 = "#38bdf8"          # Sky 400
            fg_h3 = "#a78bfa"          # Violet 400
            fg_bullet = "#10b981"      # Emerald 500
            fg_num = "#38bdf8"         # Sky 400
            bg_code_inline = "#1e293b" # Slate 800
            fg_code_inline = "#38bdf8" # Sky 400
            bg_codeblock = "#0f172a"   # Slate 900
            fg_codeblock = "#7dd3fc"   # Sky 300
            fg_quote = "#94a3b8"       # Slate 400
            fg_cursor = "#10b981"      # Emerald
            bg_widget = "#0f172a"
        else:
            fg_main = "#1e293b"        # Slate 800
            fg_secondary = "#334155"   # Slate 700
            fg_bold = "#0f172a"        # Slate 900
            fg_h1 = "#059669"          # Emerald 600
            fg_h2 = "#0284c7"          # Sky 600
            fg_h3 = "#7c3aed"          # Violet 600
            fg_bullet = "#059669"
            fg_num = "#0284c7"
            bg_code_inline = "#f1f5f9"
            fg_code_inline = "#0369a1"
            bg_codeblock = "#f8fafc"
            fg_codeblock = "#0f172a"
            fg_quote = "#64748b"
            fg_cursor = "#059669"
            bg_widget = "#ffffff"

        self.widget.configure(bg=bg_widget, insertbackground=fg_cursor)

        # Headings
        self.widget.tag_configure("h1", font=("Segoe UI", 15, "bold"), foreground=fg_h1, spacing1=10, spacing3=4)
        self.widget.tag_configure("h2", font=("Segoe UI", 13, "bold"), foreground=fg_h2, spacing1=8, spacing3=3)
        self.widget.tag_configure("h3", font=("Segoe UI", 12, "bold"), foreground=fg_h3, spacing1=6, spacing3=2)

        # Body Text
        self.widget.tag_configure("regular", font=("Segoe UI", 12), foreground=fg_main, spacing1=2, spacing3=2)
        self.widget.tag_configure("bold", font=("Segoe UI", 12, "bold"), foreground=fg_bold)
        self.widget.tag_configure("italic", font=("Segoe UI", 12, "italic"), foreground=fg_secondary)
        self.widget.tag_configure("bold_italic", font=("Segoe UI", 12, "bold italic"), foreground=fg_bold)

        # Lists
        self.widget.tag_configure("bullet_symbol", font=("Segoe UI", 12, "bold"), foreground=fg_bullet, lmargin1=16, lmargin2=32)
        self.widget.tag_configure("bullet_text", font=("Segoe UI", 12), foreground=fg_main, lmargin1=16, lmargin2=32, spacing1=2, spacing3=2)
        self.widget.tag_configure("num_symbol", font=("Segoe UI", 12, "bold"), foreground=fg_num, lmargin1=16, lmargin2=32)
        self.widget.tag_configure("num_text", font=("Segoe UI", 12), foreground=fg_main, lmargin1=16, lmargin2=32, spacing1=2, spacing3=2)

        # Code & Quotes
        self.widget.tag_configure("code_inline", font=("Consolas", 11), foreground=fg_code_inline, background=bg_code_inline)
        self.widget.tag_configure("code_block", font=("Consolas", 11), foreground=fg_codeblock, background=bg_codeblock, lmargin1=24, lmargin2=24, spacing1=4, spacing3=4)
        self.widget.tag_configure("quote", font=("Segoe UI", 12, "italic"), foreground=fg_quote, lmargin1=20, lmargin2=20)
        self.widget.tag_configure("cursor", font=("Segoe UI", 12, "bold"), foreground=fg_cursor)

    def clear(self):
        self.raw_text = ""
        self.widget.configure(state="normal")
        self.widget.delete("1.0", "end")
        self.widget.configure(state="disabled")

    def append_chunk(self, chunk: str):
        """Appends a new streaming chunk and refreshes the styled display."""
        self.raw_text += chunk
        self.render(self.raw_text, streaming=True)

    def finish_streaming(self):
        """Marks generation as completed, removing the blinking streaming cursor."""
        self.is_streaming = False
        self.render(self.raw_text, streaming=False)

    def render(self, text: str, streaming: bool = False):
        """Parses and renders formatted markdown text into the text widget."""
        self.is_streaming = streaming
        self.widget.configure(state="normal")
        self.widget.delete("1.0", "end")

        lines = text.split("\n")
        in_code_block = False

        for i, line in enumerate(lines):
            # Check for code fence ```
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self.widget.insert("end", line + "\n", "code_block")
                continue

            # Heading 1: # Title
            if line.startswith("# "):
                h_text = line[2:].strip()
                self._insert_inline_formatted(h_text, base_tag="h1")
                self.widget.insert("end", "\n")
                continue

            # Heading 2: ## Subtitle
            if line.startswith("## "):
                h_text = line[3:].strip()
                self._insert_inline_formatted(h_text, base_tag="h2")
                self.widget.insert("end", "\n")
                continue

            # Heading 3: ### Section
            if line.startswith("### "):
                h_text = line[4:].strip()
                self._insert_inline_formatted(h_text, base_tag="h3")
                self.widget.insert("end", "\n")
                continue

            # Bullet List: - item or • item or * item
            bullet_match = re.match(r"^(\s*)[•\-\*]\s+(.*)$", line)
            if bullet_match:
                indent, content = bullet_match.group(1), bullet_match.group(2)
                self.widget.insert("end", indent + "●  ", "bullet_symbol")
                self._insert_inline_formatted(content, base_tag="bullet_text")
                self.widget.insert("end", "\n")
                continue

            # Numbered List: 1. item
            num_match = re.match(r"^(\s*)(\d+[\.\)])\s+(.*)$", line)
            if num_match:
                indent, num_str, content = num_match.group(1), num_match.group(2), num_match.group(3)
                self.widget.insert("end", indent + f"{num_str}  ", "num_symbol")
                self._insert_inline_formatted(content, base_tag="num_text")
                self.widget.insert("end", "\n")
                continue

            # Blockquote: > quote
            if line.startswith("> "):
                quote_text = line[2:].strip()
                self._insert_inline_formatted(quote_text, base_tag="quote")
                self.widget.insert("end", "\n")
                continue

            # Regular paragraph line
            if line.strip():
                self._insert_inline_formatted(line, base_tag="regular")
                self.widget.insert("end", "\n")
            else:
                # Empty line spacing
                self.widget.insert("end", "\n")

        # Append animated typing cursor if streaming
        if streaming:
            self.widget.insert("end", " ▌", "cursor")

        self.widget.see("end")
        self.widget.configure(state="disabled")

    def _insert_inline_formatted(self, text: str, base_tag: str = "regular"):
        """
        Parses inline markdown tokens:
        - `code` -> code_inline
        - **bold** -> bold
        - *italic* -> italic
        - Plain text -> base_tag
        """
        # Tokenizer regex: matches `code`, **bold**, *italic*
        pattern = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
        pos = 0

        for match in pattern.finditer(text):
            start, end = match.span()
            # Text before match
            if start > pos:
                self.widget.insert("end", text[pos:start], base_tag)

            matched_str = match.group(0)
            if matched_str.startswith("`") and matched_str.endswith("`"):
                # Code snippet
                code_content = matched_str[1:-1]
                self.widget.insert("end", f" {code_content} ", "code_inline")
            elif matched_str.startswith("**") and matched_str.endswith("**"):
                # Bold text
                bold_content = matched_str[2:-2]
                self.widget.insert("end", bold_content, "bold")
            elif matched_str.startswith("*") and matched_str.endswith("*"):
                # Italic text
                italic_content = matched_str[1:-1]
                self.widget.insert("end", italic_content, "italic")

            pos = end

        # Remaining trailing text
        if pos < len(text):
            self.widget.insert("end", text[pos:], base_tag)
