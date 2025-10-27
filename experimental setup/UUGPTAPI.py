#!/usr/bin/env python3
# UUGPTAPI.py
# Usage examples are below (CMD on Windows).

import time
import sys
import argparse
from pathlib import Path
from openai import OpenAI

API_KEY = "[API KEY HERE]"   # <-- put your API key here
MODEL   = "gpt-5"                      # or "gpt-5-mini" / "gpt-5-nano"

INSTRUCTIONS = """You, a software developer, need to insert code documentation to make the Python code more readable and comprehensible. This code documentation should be done on all functions and classes to explain what they do. The documentation of a function should be written underneath the function header and the documentation of a class should be written underneath the class header. To improve the readability of a function, comments between the lines in functions might also be needed but don’t make it redundant if not necessary. As an input, you will get a Python file that doesn't contain any comments and only consists of a class and functions. You will then insert the comments into that file to complement the code. The output should be the same Python file with the same code but with the added documentation for functions and classes. Also, the output file should only be the Python file and nothing else."""

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    parts = s.split("```")
    inner = "".join(parts[1:-1]) if len(parts) >= 2 else s
    if inner.lstrip().lower().startswith("python"):
        inner = inner.split("\n", 1)[1] if "\n" in inner else ""
    return inner.strip()

def build_prompt_zero(target_name: str, target_code: str) -> str:
    return (
        f"{INSTRUCTIONS}\n\n"
        f"<TARGET name='{target_name}'>\n{target_code}\n</TARGET>\n\n"
        f"Return only the fully commented Python code for {target_name}."
    )

def build_prompt_few(target_name: str, target_code: str, examples: list[tuple[str, str]]) -> str:
    # examples is a list of (name, code)
    examples_blob = "\n\n".join(
        f"# --- BEGIN EXAMPLE {i}: {name} ---\n{code}\n# --- END EXAMPLE {i}: {name} ---"
        for i, (name, code) in enumerate(examples, start=1)
    )
    return (
        f"{INSTRUCTIONS}\n\n"
        f"<EXAMPLES>\n{examples_blob}\n</EXAMPLES>\n\n"
        f"<TARGET name='{target_name}'>\n{target_code}\n</TARGET>\n\n"
        f"Return only the fully commented Python code for {target_name}."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Add comments/docstrings to a Python file using GPT-5."
    )
    parser.add_argument("--mode", choices=["zero", "few"], default="zero",
                        help="zero: no examples; few: include example files to learn style.")
    parser.add_argument("target", help="Path to the Python file to comment.")
    parser.add_argument("examples", nargs="*", help="(few-shot only) One or more commented example .py files.")
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        raise SystemExit(f"Target not found: {target_path}")

    target_code = read_text(target_path)

    if args.mode == "few":
        if not args.examples:
            # Gracefully fall back to zero-shot if no examples were provided.
            prompt = build_prompt_zero(target_path.name, target_code)
        else:
            ex_list = []
            for p in args.examples:
                ex_path = Path(p)
                if not ex_path.exists():
                    raise SystemExit(f"Example not found: {ex_path}")
                ex_list.append((ex_path.name, read_text(ex_path)))
            prompt = build_prompt_few(target_path.name, target_code, ex_list)
    else:
        prompt = build_prompt_zero(target_path.name, target_code)

    client = OpenAI(api_key=API_KEY)

    t0 = time.perf_counter()
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": "minimal"},   # GPT-5: keep output concise; no temperature on reasoning models
    )
    elapsed = time.perf_counter() - t0

    print(strip_code_fences(resp.output_text))  # ONLY the new Python code
    print(f"[timing] total_seconds={elapsed:.3f}", file=sys.stderr)

if __name__ == "__main__":
    main()
