#!/usr/bin/env python3
"""
run_my_slice.py - score a model on a slice you downloaded from the
Medical Benchmark Disease Index.

This is the piece Joon asked for on Sep 25: filter to your disease area,
download the CSV, run a model on it yourself, get a number.

USAGE
    pip install pandas requests
    python run_my_slice.py my_slice.csv --limit 20

    Default is a local Ollama model, so it is free and needs no key.
    Hosted instead:
        export OPENAI_API_KEY=sk-...
        python run_my_slice.py my_slice.csv --backend openai --model gpt-4o-mini

IMPORTANT
    Prompt from `answer_options_plain`, never `answer_options`. The second
    column marks the right choice "[expected]" so a human can read it, so a
    model prompted with it scores near 100% by leakage. This script uses the
    plain column when present and strips the tags when it is not.

WHAT IT SKIPS
    Items whose question text is withheld for licensing, items with no
    answer options, and open-ended items with no single keyed answer.
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests

PROMPT = (
    "Answer this multiple-choice medical question.\n"
    "Reply with the letter of the single best option and nothing else.\n\n"
    "{question}\n\n{options}"
)

TAG = re.compile(r"\s*\[(?:expected|incorrect)(?:;[^\]]*)?\]\s*$")
WITHHELD = "[Question text withheld"
NO_OPTIONS = "[Answer options unavailable"


def clean_options(item):
    """Options with no [expected]/[incorrect] markers.

    Prefers the answer_options_plain column. Falls back to stripping the
    markers off answer_options, for CSVs downloaded before that column
    existed.
    """
    plain = item.get("answer_options_plain")
    if isinstance(plain, str) and plain.strip():
        return plain
    cell = item.get("answer_options")
    if not isinstance(cell, str):
        return ""
    return "\n".join(TAG.sub("", line).rstrip()
                     for line in cell.splitlines() if line.strip())


def expected_letter(cell):
    """'C - Reassurance and follow-up' -> 'C'. Returns None if not lettered."""
    if not isinstance(cell, str):
        return None
    m = re.match(r"\s*([A-J])\s*[·.:\-]\s", cell)
    return m.group(1) if m else None


def picked_letter(reply):
    if not reply:
        return None
    m = re.search(r"\b([A-J])\b", reply.strip().upper())
    return m.group(1) if m else None


def tested_conditions(cell):
    """'short stature [tested]; obesity [context]' -> ['short stature']"""
    if not isinstance(cell, str):
        return []
    out = []
    for part in cell.split(";"):
        part = part.strip()
        if part.endswith("[tested]"):
            out.append(part[: -len("[tested]")].strip())
    return out


def ask_ollama(prompt, model, host):
    r = requests.post(f"{host}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0}},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")


def ask_openai(prompt, model, key):
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]},
                      timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="the CSV you downloaded from the Space")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--limit", type=int, default=0, help="stop after N items")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    need = {"item_id", "source", "question", "expected_answer", "answer_options"}
    if not need.issubset(df.columns):
        sys.exit(f"Missing columns: {sorted(need - set(df.columns))}\n"
                 f"Found: {list(df.columns)}\n"
                 f"This script expects the Space's 'Download filtered CSV' format.")

    total = len(df)
    df = df[~df["question"].astype(str).str.startswith(WITHHELD)]
    withheld = total - len(df)

    before = len(df)
    df = df[~df["answer_options"].astype(str).str.startswith(NO_OPTIONS)]
    no_options = before - len(df)

    df = df.copy()
    df["expected"] = df["expected_answer"].map(expected_letter)
    before = len(df)
    df = df[df["expected"].notna()]
    unlettered = before - len(df)

    if args.limit:
        df = df.head(args.limit)

    print(f"{total} items in the slice. Scoring {len(df)}.")
    print(f"  skipped {withheld} with question text withheld, "
          f"{no_options} with no answer options, "
          f"{unlettered} with no single lettered answer.")
    if df.empty:
        sys.exit("Nothing to score.")

    key = os.environ.get("OPENAI_API_KEY") if args.backend == "openai" else None
    if args.backend == "openai" and not key:
        sys.exit("Set OPENAI_API_KEY first.")

    rows = []
    for n, (_, item) in enumerate(df.iterrows(), 1):
        prompt = PROMPT.format(question=str(item["question"]).strip(),
                               options=clean_options(item))
        try:
            reply = (ask_ollama(prompt, args.model, args.host)
                     if args.backend == "ollama"
                     else ask_openai(prompt, args.model, key))
        except Exception as e:                       # noqa: BLE001
            reply = f"[ERROR] {e}"
            time.sleep(2)
        pick = picked_letter(reply)
        rows.append({"item_id": item["item_id"], "source": item["source"],
                     "expected": item["expected"], "picked": pick,
                     "correct": pick == item["expected"],
                     "conditions": item.get("conditions", ""),
                     "reply": str(reply)[:500]})
        if n % 10 == 0 or n == len(df):
            print(f"  {n}/{len(df)}")

    out = pd.DataFrame(rows)
    stem = os.path.splitext(args.csv)[0]
    out.to_csv(f"{stem}_responses.csv", index=False)

    lines = [
        f"Model:      {args.model} via {args.backend}",
        f"Slice:      {args.csv}",
        f"Scored:     {len(out)} of {total} items",
        f"Accuracy:   {out['correct'].mean():.1%}",
        f"Unparsable: {int(out['picked'].isna().sum())} replies had no clear "
        f"letter (counted as wrong)",
        "",
        "By source benchmark:",
    ]
    for src, r in out.groupby("source")["correct"].agg(["mean", "count"]).iterrows():
        lines.append(f"  {src:<16} {r['mean']:.1%}  (n={int(r['count'])})")

    exploded = out.assign(cond=out["conditions"].map(tested_conditions)).explode("cond")
    exploded = exploded[exploded["cond"].notna()]
    if len(exploded):
        by_cond = exploded.groupby("cond")["correct"].agg(["mean", "count"])
        by_cond = by_cond[by_cond["count"] >= 5].sort_values("mean")
        if len(by_cond):
            lines += ["", "By tested condition (5 or more items), worst first:"]
            for cond, r in by_cond.iterrows():
                lines.append(f"  {str(cond)[:40]:<42} {r['mean']:.1%}  "
                             f"(n={int(r['count'])})")

    summary = "\n".join(lines)
    with open(f"{stem}_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nWrote {stem}_responses.csv and {stem}_summary.txt")


if __name__ == "__main__":
    main()
