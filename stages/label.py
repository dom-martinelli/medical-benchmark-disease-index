from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .common import (
    CHECKPOINTS,
    DATA,
    PROMPTS,
    append_jsonl,
    append_log,
    ensure_dirs,
    json_for_prompt,
    log_prompt,
    prompt_version,
    read_jsonl,
    write_parquet_atomic,
)
from .models import ModelClient, choose_primary_model

PROMPT_PATH = PROMPTS / "label_conditions.md"
CHECKPOINT = CHECKPOINTS / "labels.jsonl"
ROLES = {"tested", "context", "ruled_out", "wrong_option"}
CATEGORIES = {"ethics or law", "statistics or epidemiology", "basic science", "health systems", "communication", "other"}


def _normalize_response(obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw_conditions = obj.get("conditions") or []
    if not isinstance(raw_conditions, list):
        raise ValueError("conditions must be a list")
    rows: list[dict[str, Any]] = []
    for condition in raw_conditions:
        if not isinstance(condition, dict) or not str(condition.get("condition_name", "")).strip():
            continue
        quote = " ".join(str(condition.get("supporting_quote", "")).split()[:12])
        role = str(condition.get("role", "")).strip().casefold()
        confidence = str(condition.get("confidence", "low")).strip().casefold()
        rows.append(
            {
                "condition_name": str(condition["condition_name"]).strip(),
                "role": role if role in ROLES else "context",
                "supporting_quote": quote,
                "confidence": confidence if confidence in {"high", "low"} else "low",
                "no_condition_category": "",
            }
        )
    if not rows:
        category = str(obj.get("no_condition_category") or "other").strip().casefold()
        rows.append(
            {
                "condition_name": "",
                "role": "",
                "supporting_quote": "",
                "confidence": "",
                "no_condition_category": category if category in CATEGORIES else "other",
            }
        )
    return rows


FAILED = "label_failed"


def label_one(item: dict[str, Any], client: ModelClient, prompt_template: str) -> list[dict[str, Any]]:
    rendered = prompt_template.replace("{{ITEM_JSON}}", json_for_prompt(item))
    last_error: Exception | None = None
    attempt = 0
    outage_waited = 0.0
    while attempt < 3:
        try:
            obj = client.generate_json(rendered)
            return _normalize_response(obj)
        except requests.ConnectionError as exc:
            # The model server is down (for example the Mac slept). Wait for it
            # for up to 30 minutes rather than burning the item's attempts.
            last_error = exc
            if outage_waited >= 1800:
                break
            time.sleep(30)
            outage_waited += 30
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Model failed to return the required JSON after 3 attempts: {last_error}")


def failed_row(exc: Exception) -> dict[str, Any]:
    return {
        "condition_name": "",
        "role": "",
        "supporting_quote": "",
        "confidence": "",
        "no_condition_category": FAILED,
        "label_error": f"{type(exc).__name__}: {exc}"[:300],
    }


def keep_latest_outcome(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop failure markers for items that were later labeled successfully."""
    succeeded = {str(r["item_id"]) for r in rows if r.get("no_condition_category") != FAILED}
    output, failed_seen = [], set()
    for row in rows:
        item_id = str(row["item_id"])
        if row.get("no_condition_category") == FAILED:
            if item_id in succeeded or item_id in failed_seen:
                continue
            failed_seen.add(item_id)
        output.append(row)
    return output


def _matching_checkpoint_rows(
    valid_ids: set[str], model_name: str, settings: str, version: str
) -> list[dict[str, Any]]:
    return [
        row for row in read_jsonl(CHECKPOINT)
        if str(row.get("item_id")) in valid_ids
        and row.get("model_name") == model_name
        and row.get("model_settings") == settings
        and row.get("prompt_version") == version
    ]


def _write_outputs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        write_parquet_atomic(df, DATA / "labels.parquet")
    return df


def _write_pilot(items: pd.DataFrame, labels: pd.DataFrame) -> None:
    pilot_ids = list(dict.fromkeys(labels["item_id"].tolist()))[:50]
    subset = labels[labels["item_id"].isin(pilot_ids)]
    grouped = {
        item_id: "; ".join(
            f"{r.condition_name or '[' + r.no_condition_category + ']'} ({r.role or 'no condition'}, {r.confidence or 'n/a'})"
            for r in frame.itertuples()
        )
        for item_id, frame in subset.groupby("item_id", sort=False)
    }
    preview = items[items["item_id"].isin(pilot_ids)][["item_id", "source", "question"]].copy()
    preview["question"] = preview["question"].str.replace("\n", " ").str.replace("|", "\\|", regex=False)
    preview["labels"] = preview["item_id"].map(grouped)
    preview["labels"] = preview["labels"].str.replace("|", "\\|", regex=False)
    text = "# Fifty-item condition-label pilot\n\n" + preview.to_markdown(index=False) + "\n"
    (DATA / "pilot_labels.md").write_text(text, encoding="utf-8")
    print(text)


def run(approve_pilot: bool = False, allow_paid: bool = False, allow_pull: bool = True) -> bool:
    ensure_dirs()
    if not (DATA / "items.parquet").exists():
        raise FileNotFoundError("Stage 1 output data/items.parquet is missing. Run collection first.")
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    log_prompt(PROMPT_PATH, "Stage 2")
    client, reason = choose_primary_model(allow_paid=allow_paid, allow_pull=allow_pull)
    append_log(
        "Stage 2 model choice",
        f"Model: `{client.provider}/{client.model_name}`  \nSettings: `{json.dumps(client.settings, sort_keys=True)}`  \n{reason}",
    )
    items = pd.read_parquet(DATA / "items.parquet")
    model_full = f"{client.provider}/{client.model_name}"
    settings_text = json.dumps(client.settings, sort_keys=True)
    version = prompt_version(PROMPT_PATH)
    valid_ids = set(items["item_id"].astype(str))
    matching_rows = _matching_checkpoint_rows(valid_ids, model_full, settings_text, version)
    matching_rows = keep_latest_outcome(matching_rows)
    # Items that failed on an earlier run are retried once per run.
    completed = {str(x["item_id"]) for x in matching_rows if x.get("no_condition_category") != FAILED}
    pilot = items.groupby("source", sort=False, group_keys=False).head(5).head(50)
    if len(pilot) < min(50, len(items)):
        extra = items[~items["item_id"].isin(pilot["item_id"])].head(50 - len(pilot))
        pilot = pd.concat([pilot, extra], ignore_index=True)
    pilot_target = len(pilot)
    ordered = pd.concat([pilot, items[~items["item_id"].isin(pilot["item_id"])]], ignore_index=True)
    selected = ordered if approve_pilot else pilot
    todo = [row._asdict() for row in selected.itertuples(index=False) if row.item_id not in completed]
    bench_started = time.monotonic()

    failed_now: set[str] = set()
    for index, item in enumerate(todo, start=1):
        try:
            labels = label_one(item, client, prompt_text)
        except Exception as exc:
            labels = [failed_row(exc)]
            failed_now.add(item["item_id"])
            print(f"Could not label {item['item_id']}; recorded it as label_failed and moved on.")
        common = {
            "item_id": item["item_id"],
            "source": item["source"],
            "model_name": model_full,
            "model_settings": settings_text,
            "prompt_text": prompt_text,
            "prompt_version": version,
            "run_date": date.today().isoformat(),
        }
        new_rows = [{**common, **row} for row in labels]
        append_jsonl(CHECKPOINT, new_rows)
        matching_rows.extend(new_rows)
        if item["item_id"] not in failed_now:
            completed.add(item["item_id"])
        total = len(completed) + len(failed_now)
        if index == 20:
            elapsed = time.monotonic() - bench_started
            rate = 20 / elapsed * 60
            append_log("Stage 2 twenty-item speed benchmark", f"Model: `{client.model_name}`  \nItems per minute: {rate:.2f}  \nElapsed seconds: {elapsed:.1f}")
            print(f"Twenty-item speed benchmark: {rate:.2f} items per minute with {client.model_name}.")
        if total % 100 == 0:
            per_item = (time.monotonic() - bench_started) / index
            hours_left = per_item * (len(todo) - index) / 3600
            print(f"Labeled {total:,} of {len(selected):,} items so far, about {hours_left:.1f} hours left.")

    matching_rows = keep_latest_outcome(matching_rows)
    labels_df = _write_outputs(matching_rows)
    if len(completed | failed_now) >= pilot_target and not approve_pilot and len(items) > pilot_target:
        _write_pilot(items, labels_df)
        append_log("Stage 2 pilot pause", f"Pilot items completed: {pilot_target}. Waiting for Dominic's explicit go before the full run.")
        print("Stage 2 paused after the required 50-item pilot. Review data/pilot_labels.md and rerun with --approve-pilot after Dominic says go.")
        return False
    failed_ids = {str(r["item_id"]) for r in matching_rows if r.get("no_condition_category") == FAILED}
    if len(completed | failed_ids) < len(items):
        raise RuntimeError(f"Stage 2 ended with only {len(completed | failed_ids):,} of {len(items):,} items checkpointed.")
    append_log(
        "Stage 2 complete",
        f"Items labeled: {len(completed):,}  \nItems that failed after retries (kept, flagged label_failed): {len(failed_ids):,}  \n"
        f"Output rows: {len(labels_df):,}",
    )
    print(f"Stage 2 complete: wrote {len(labels_df):,} label rows for {len(completed):,} items.")
    return True
