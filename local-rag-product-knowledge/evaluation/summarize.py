#!/usr/bin/env python3
"""Print a Markdown summary table of all test runs in evaluation/results/.

Reads only the run_info_*.json files written by `rag.py test`, so the numbers
in the README can be checked by anyone:  python evaluation/summarize.py
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    rows = []
    for f in sorted(RESULTS.glob("*/run_info_*.json")):
        info = json.loads(f.read_text(encoding="utf-8"))
        rag = info["mode"] == "rag"
        order = info["settings"]["top_k"] if rag else 0
        rows.append((order,
            f.parent.name,
            f"RAG, top-k {info['settings']['top_k']}" if rag else "No RAG (baseline)",
            info["n_questions"],
            info["expected_page_retrieved"] if rag else "n/a",
            info["mean_total_s"],
            info["max_total_s"],
        ))
    rows.sort(key=lambda r: (r[1], r[0]))
    print("| Manual | Mode | Questions | Expected page retrieved | Mean s / question | Max s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r[1:]) + " |")


if __name__ == "__main__":
    main()
