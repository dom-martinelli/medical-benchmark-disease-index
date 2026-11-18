"""Reproduce the Space's 'Download filtered CSV' output without loading gradio.
Mirrors app.filter_rows exactly. Usage: python make_slice.py "stature" out.csv
"""
import json, sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODED = pd.read_parquet(ROOT / "data" / "explorer_coded.parquet")
ITEMS = pd.read_parquet(ROOT / "data" / "explorer_items.parquet")

query, out = sys.argv[1], sys.argv[2]
needle = query.strip().casefold()
rows = CODED[
    CODED["condition_name"].fillna("").str.casefold().str.contains(needle, regex=False)
    | CODED["ccsr_description"].fillna("").str.casefold().str.contains(needle, regex=False)
    | CODED["code"].fillna("").str.casefold().str.contains(needle, regex=False)
]

merged = rows.merge(
    ITEMS[["item_id", "question", "expected_answer_key", "expected_answer",
           "answer_options", "wrong_option_map",
           "question_redistribution_allowed", "license"]],
    on="item_id", how="left", suffixes=("", "_item"),
)
merged["condition_display"] = merged.apply(
    lambda x: f"{x['condition_name']} [{x['role']}]" if x["condition_name"]
    else f"[{x['no_condition_category']}]", axis=1)
merged["code_display"] = merged.apply(
    lambda x: f"{x['code']} · {x['ccsr_description']}" if x["code"] != "unmatched"
    else "unmatched", axis=1)

grouped = merged.groupby(["item_id", "source"], sort=False, as_index=False).agg(
    question=("question", "first"),
    expected_answer_key=("expected_answer_key", "first"),
    expected_answer=("expected_answer", "first"),
    answer_options=("answer_options", "first"),
    wrong_option_map=("wrong_option_map", "first"),
    allowed=("question_redistribution_allowed", "first"),
    license=("license", "first"),
    conditions=("condition_display", lambda x: "; ".join(dict.fromkeys(x))),
    codes=("code_display", lambda x: "; ".join(dict.fromkeys(x))),
)
grouped["license_note"] = grouped.apply(
    lambda x: f"Question shown; source card license: {x['license']}" if x["allowed"]
    else f"Question withheld; source card license is {x['license'] or 'unknown'}", axis=1)
grouped.loc[~grouped["allowed"], "question"] = \
    "[Question text withheld because redistribution permission is unclear.]"
grouped["expected_answer"] = grouped.apply(
    lambda x: f"{x['expected_answer_key']} · {x['expected_answer']}"
    if x["expected_answer_key"]
    and x["expected_answer_key"].casefold() != x["expected_answer"].casefold()
    else x["expected_answer"], axis=1)


def display_options(row):
    try:
        options = json.loads(row["answer_options"] or "{}")
        wrong_map = json.loads(row["wrong_option_map"] or "{}")
    except (TypeError, ValueError):
        return "[Answer options unavailable.]"
    if not options:
        return "[Answer options unavailable or withheld.]"
    lines = []
    for key, value in options.items():
        status = "expected" if str(key) == str(row["expected_answer_key"]) else "incorrect"
        flagged = wrong_map.get(str(key), [])
        suffix = f"; wrong-option conditions: {', '.join(flagged)}" if flagged else ""
        lines.append(f"{key}. {value} [{status}{suffix}]")
    return "\n".join(lines)


def plain_options(row):
    try:
        options = json.loads(row["answer_options"] or "{}")
    except (TypeError, ValueError):
        return ""
    return "\n".join(f"{k}. {v}" for k, v in options.items()) if options else ""


grouped["answer_options_plain"] = grouped.apply(plain_options, axis=1)
grouped["answer_options"] = grouped.apply(display_options, axis=1)
table = grouped[["item_id", "source", "question", "expected_answer_key", "expected_answer",
                 "answer_options", "answer_options_plain",
                 "conditions", "codes", "license_note"]]
table.to_csv(out, index=False)
print(f"{len(table)} items written to {out}")
