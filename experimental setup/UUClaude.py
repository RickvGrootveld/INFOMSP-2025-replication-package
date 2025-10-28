# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# UUClaude.py - Comment generator with zero-shot and few-shot modes for Claude.
# Usage examples:
#   ZERO-SHOT: python UUClaude.py --mode zero  target.py --out target_annotated.py
#   FEW-SHOT:  python UUClaude.py --mode few   target.py practice1.py practice2.py practice3.py --out target_annotated.py
#
# Optional:
#   --system-file my_system.txt        # replace system prompt
#   --user-prompt-file my_user.txt     # extra user instructions injected before files
#   --model claude-sonnet-4-5          # change model
#   --max-tokens 4096                  # change output cap
#   --temperature 0.2                  # sampling temperature

import os
import sys
import time
from datetime import datetime
import argparse
from typing import List
import anthropic

# HARD-CODED API KEY (replace this)
API_KEY = "[API_KEY-HERE]"
if API_KEY == "sk-ant-REPLACE_ME":
    sys.exit("ERROR: Replace API_KEY with your real Anthropic key before running.")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def default_system_prompt(mode: str) -> str:
    base = (
        "You are a senior Python documentation assistant.\n"
        "TASK: Insert high-quality comments and docstrings into a Python source file.\n"
        "HARD RULES:\n"
        " - Do NOT change behavior, names, signatures, imports, or module structure.\n"
        " - Only add comments/docstrings and minimal blank lines where needed.\n"
        " - Return ONLY the complete modified TARGET source code as plain text.\n"
        " - Do NOT include backticks, markdown fences, or any prose.\n"
    )
    if mode == "zero":
        extra = (
            "MODE: ZERO-SHOT.\n"
            "No examples are provided. Use best practices for clear, concise Python comments and docstrings.\n"
            "Prefer the dominant style already present in the file if any; otherwise use Google-style docstrings.\n"
        )
    else:
        extra = (
            "MODE: FEW-SHOT PRACTICE.\n"
            "You will see three PRACTICE files. Internally infer a consistent commenting style from them "
            "(tone, density, docstring format, inline comment conventions). Do NOT output the practice files. "
            "Apply the learned style to the TARGET and return only the TARGET code.\n"
        )
    return base + extra

def build_user_blocks_zero(target_path: str, user_prompt: str) -> List[dict]:
    blocks: List[dict] = []
    if user_prompt:
        blocks.append({"type": "text", "text": user_prompt})
    blocks.append({
        "type": "text",
        "text": (
            "Generate comments and docstrings for the TARGET file below.\n"
            "Return only the fully rewritten TARGET as plain text.\n"
        )
    })
    blocks.append({
        "type": "text",
        "text": f"<TARGET filename='{os.path.basename(target_path)}'>\n{read_text(target_path)}\n</TARGET>"
    })
    blocks.append({"type": "text", "text": "Output only the modified TARGET source code. No fences. No extra text."})
    return blocks

def build_user_blocks_few(target_path: str, practice_paths: List[str], user_prompt: str) -> List[dict]:
    blocks: List[dict] = []
    if user_prompt:
        blocks.append({"type": "text", "text": user_prompt})
    blocks.append({
        "type": "text",
        "text": (
            "Practice internally on the following PRACTICE files to infer comment density, tone, and formatting.\n"
            "Do NOT output these. Then apply the learned style to the TARGET and return only the TARGET.\n"
        )
    })
    for i, p in enumerate(practice_paths, start=1):
        blocks.append({
            "type": "text",
            "text": f"<PRACTICE id='{i}' filename='{os.path.basename(p)}'>\n{read_text(p)}\n</PRACTICE>"
        })
    blocks.append({
        "type": "text",
        "text": f"<TARGET filename='{os.path.basename(target_path)}'>\n{read_text(target_path)}\n</TARGET>"
    })
    blocks.append({"type": "text", "text": "Return only the modified TARGET source code. No fences. No extra text."})
    return blocks

def main():
    parser = argparse.ArgumentParser(description="Generate code comments with Claude in zero-shot or few-shot practice mode.")
    parser.add_argument("--mode", choices=["zero", "few"], required=True,
                        help="zero = 1 file (target only); few = 1 target + 3 practice files")
    parser.add_argument("files", nargs="+",
                        help="Files: for zero, 1 path (TARGET). For few, 4 paths: TARGET then 3 PRACTICE.")
    parser.add_argument("--out", default=None, help="Write the annotated code to this path (prints nothing if set)")
    parser.add_argument("--model", default="claude-sonnet-4", help="Anthropic model name")
    parser.add_argument("--max-tokens", type=int, default=40960, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--system-file", default=None, help="Path to a custom system prompt file")
    parser.add_argument("--user-prompt-file", default=None, help="Path to a custom user prompt file injected before files")
    args = parser.parse_args()

    # Validate file counts by mode
    if args.mode == "zero":
        if len(args.files) != 1:
            sys.exit("ERROR: zero-shot mode requires exactly 1 file: TARGET")
        target = args.files[0]
        practice = []
    else:
        if len(args.files) != 4:
            sys.exit("ERROR: few-shot mode requires exactly 4 files: TARGET then 3 PRACTICE files")
        target = args.files[0]
        practice = args.files[1:]

    # Build prompts
    system_prompt = read_text(args.system_file) if args.system_file else default_system_prompt(args.mode)
    user_prompt = read_text(args.user_prompt_file) if args.user_prompt_file else ""

    if args.mode == "zero":
        content_blocks = build_user_blocks_zero(target, user_prompt)
    else:
        content_blocks = build_user_blocks_few(target, practice, user_prompt)

    # Create client BEFORE requests
    client = anthropic.Anthropic(api_key=API_KEY)

    # timing + request kwargs
    start_ts = datetime.now().isoformat(timespec="milliseconds")
    t0 = time.perf_counter()

    msg_kwargs = dict(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": content_blocks}],
    )

    text_result = None
    used_streaming = False

    try:
        # First try non-streaming
        resp = client.messages.create(**msg_kwargs)
        out_text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                out_text_parts.append(block.text)
        text_result = "".join(out_text_parts)
    except ValueError as e:
        # Auto-fallback if Anthropic estimates >10 min and requires streaming
        if "Streaming is required for operations that may take longer than 10 minutes" in str(e):
            used_streaming = True
            text_chunks = []
            with client.messages.stream(**msg_kwargs) as stream:
                for chunk in stream.text_stream:
                    text_chunks.append(chunk)
            text_result = "".join(text_chunks)
        else:
            raise

    if text_result is None:
        raise RuntimeError("No text received from Claude.")

    t1 = time.perf_counter()
    end_ts = datetime.now().isoformat(timespec="milliseconds")
    elapsed_ms = (t1 - t0) * 1000.0

    # Log timing to STDERR so stdout remains code-only
    print(
        f"[TIMING] start={start_ts} end={end_ts} elapsed_ms={elapsed_ms:.1f} mode={'stream' if used_streaming else 'nonstream'}",
        file=sys.stderr,
    )

    # Clean and finalize output text
    final_text = strip_code_fences(text_result).rstrip() + "\n"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(final_text)
    else:
        sys.stdout.write(final_text)

if __name__ == "__main__":
    main()
