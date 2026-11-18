from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHECKPOINTS = DATA / "checkpoints"
PROMPTS = ROOT / "prompts"
REVIEW = ROOT / "review"
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
RUN_LOG = ROOT / "run_log.md"


def ensure_dirs() -> None:
    for path in (DATA, CHECKPOINTS, REVIEW, REPORT, FIGURES, DATA / "reference"):
        path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_version(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return first.rsplit("version", 1)[-1].strip() if "version" in first else sha256_text(path.read_text())[:12]


def append_log(title: str, body: str) -> None:
    ensure_dirs()
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {now_iso()} - {title}\n\n{body.rstrip()}\n")


def log_prompt(path: Path, stage: str) -> None:
    text = path.read_text(encoding="utf-8")
    append_log(
        f"{stage} prompt snapshot",
        f"Prompt file: `{path.relative_to(ROOT)}`  \n"
        f"Prompt SHA-256: `{sha256_text(text)}`\n\n```text\n{text}\n```",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temp, index=False)
    temp.replace(path)


def load_metrics() -> dict[str, Any]:
    path = DATA / "run_metrics.json"
    if not path.exists():
        return {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "wall_seconds": 0.0}
    return json.loads(path.read_text(encoding="utf-8"))


def update_metrics(**increments: float) -> dict[str, Any]:
    metrics = load_metrics()
    for key, value in increments.items():
        metrics[key] = metrics.get(key, 0) + value
    path = DATA / "run_metrics.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return metrics


def print_cost_report(started: float) -> None:
    metrics = update_metrics(wall_seconds=time.monotonic() - started)
    print(
        "Cost report: "
        f"{int(metrics['api_calls']):,} API calls, "
        f"{int(metrics['input_tokens']):,} input tokens, "
        f"{int(metrics['output_tokens']):,} output tokens, "
        f"${float(metrics['cost_usd']):.4f}; "
        f"cumulative wall time {float(metrics['wall_seconds']) / 60:.1f} minutes."
    )


def json_for_prompt(item: dict[str, Any]) -> str:
    payload = {
        "item_id": item.get("item_id"),
        "question": item.get("question", ""),
        "options": item.get("options", "[]"),
        "answer_key": item.get("answer_key", ""),
        "explanation": item.get("explanation", ""),
    }
    return json.dumps(payload, ensure_ascii=False)
