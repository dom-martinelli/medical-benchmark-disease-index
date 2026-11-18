#!/usr/bin/env python3
"""Resolve a benchmark-index HealthBench item ID to its source record.

The public explorer intentionally withholds HealthBench conversation text. Its
displayed item ID is a deterministic hash of the original HealthBench
``prompt_id``. This script scans the pinned source JSONL, reproduces that hash,
and reports the matching source row and prompt ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request


REVISION = "40ee1968852fc57f625934251ac22be47077a8fb"
FILE_NAME = "2025-05-07-06-14-12_oss_eval.jsonl"
SOURCE_URL = (
    "https://huggingface.co/datasets/openai/healthbench/resolve/"
    f"{REVISION}/{FILE_NAME}"
)


def index_item_id(prompt_id: str) -> str:
    basis = f"HealthBench|test|{prompt_id}"
    digest = hashlib.sha256(basis.encode()).hexdigest()[:16]
    return f"healthbench-{digest}"


def resolve(item_id: str) -> tuple[int, dict]:
    if not item_id.startswith("healthbench-"):
        raise ValueError("Expected an item ID beginning with 'healthbench-'.")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "benchmark-disease-index/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        for row_number, raw_line in enumerate(response, start=1):
            row = json.loads(raw_line)
            prompt_id = str(row.get("prompt_id", ""))
            if prompt_id and index_item_id(prompt_id) == item_id:
                return row_number, row
    raise LookupError(f"No source record matched {item_id} at revision {REVISION}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a public explorer HealthBench item ID to its source row."
    )
    parser.add_argument("item_id", help="For example: healthbench-68ebc53be307db72")
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the source prompt locally. Do not republish it in the public explorer.",
    )
    args = parser.parse_args()

    try:
        row_number, row = resolve(args.item_id)
    except (ValueError, LookupError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    prompt_id = str(row["prompt_id"])
    viewer_url = (
        "https://huggingface.co/datasets/openai/healthbench/viewer/"
        f"default/test?row={row_number - 1}"
    )
    print(f"index_item_id: {args.item_id}")
    print(f"prompt_id: {prompt_id}")
    print(f"source_row_1_based: {row_number}")
    print(f"source_revision: {REVISION}")
    print(f"source_file: {SOURCE_URL}")
    print(f"dataset_viewer: {viewer_url}")

    if args.show_text:
        print("\nLocal source prompt:")
        print(json.dumps(row.get("prompt", []), ensure_ascii=False, indent=2))
        print("\nDo not copy this text into the public explorer.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
