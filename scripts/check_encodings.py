"""Scan repository files for UTF-8 decode errors or UTF-8 replacement bytes (U+FFFD).

Prints files that contain decoding errors or the UTF-8 replacement byte sequence.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXCLUDE_DIRS = {".git", "venv", ".venv", "outputs", "RAG_data", "chroma_db"}

problem_files = []

for p in ROOT.rglob("*"):
    if p.is_dir():
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        continue
    if any(part in EXCLUDE_DIRS for part in p.parts):
        continue
    # skip binary-like extensions
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".db", ".sqlite3", ".safetensors", ".pt", ".pth", ".bin", ".exe"}:
        continue
    try:
        data = p.read_bytes()
    except Exception as e:
        print(f"SKIP (read error): {p} -> {e}")
        continue
    # check for replacement bytes EF BF BD
    if b"\xef\xbf\xbd" in data:
        problem_files.append((p, "contains U+FFFD utf8 replacement bytes"))
        continue
    # try utf-8 decode
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        problem_files.append((p, f"utf-8 decode error: {e}"))

if not problem_files:
    print("No encoding problems found.")
    sys.exit(0)

print("Encoding issues found:")
for p, reason in problem_files:
    print(f" - {p}: {reason}")

# exit non-zero to indicate issues
sys.exit(2)
