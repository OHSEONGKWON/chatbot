"""Inspect and (safely) clean files containing U+FFFD replacement chars.

- Backs up original file to <file>.bak
- Writes cleaned copy where U+FFFD is replaced with a configurable marker (default: '' i.e., removed)
- Prints lines containing the replacement for manual review
"""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
TARGET = ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl"

if not TARGET.exists():
    print(f"Target not found: {TARGET}")
    sys.exit(1)

bak = TARGET.with_suffix(TARGET.suffix + '.bak')
if not bak.exists():
    TARGET.replace(bak)
    print(f"Backed up original to: {bak}")
else:
    print(f"Backup already exists: {bak}")

# read with utf-8 replace to show positions
raw = bak.read_bytes()
try:
    text = raw.decode('utf-8')
except Exception:
    text = raw.decode('utf-8', errors='replace')

lines = text.splitlines()

problem_lines = []
for i, line in enumerate(lines, 1):
    if '\ufffd' in line:
        problem_lines.append((i, line))

print(f"Found {len(problem_lines)} lines containing U+FFFD replacement chars")
for ln, line in problem_lines[:20]:
    snippet = line[:200]
    print(f"Line {ln}: {snippet}")

# create cleaned copy (remove U+FFFD)
cleaned = text.replace('\ufffd', '')
TARGET.write_text(cleaned, encoding='utf-8')
print(f"Wrote cleaned file to: {TARGET}")

if problem_lines:
    print('NOTE: removed U+FFFD from lines above; check the .bak file if manual repair is needed.')
else:
    print('No replacement chars found; nothing changed.')
