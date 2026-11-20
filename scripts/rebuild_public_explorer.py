#!/usr/bin/env python3
"""Rebuild the public explorer with harmonized item attributes and LiveClin.

The script starts from the redacted public Parquet files, removes any prior
LiveClin rows, and then adds the pinned public LiveClin release. Candidate
benchmarks remain catalog entries until their item files can be imported under
their access terms.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "config"
STATIC_DATA = ROOT / "docs" / "demo" / "data" / "explorer.json"

LIVECLIN_REVISION = "9f3199a3e76a7e86eb935d9d82ab1c589bb4da58"
LIVECLIN_PARQUET = (
    "https://huggingface.co/datasets/AQ-MedAI/LiveClin/resolve/"
    "refs%2Fconvert%2Fparquet/2025_H1/test/0000.parquet"
)

ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
    9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI", 22: "XXII",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_defaults(source: str, defaults: dict[str, dict[str, str]]) -> dict[str, str]:
    return defaults.get(source, {
        "clinical_task": "other",
        "care_setting": "unknown",
        "user_role": "unknown",
        "response_format": "unknown",
        "evaluation_method": "benchmark-native metric",
        "reference_standard": "benchmark-native reference",
        "harmonization_provenance": "source-level rule",
    })


def _liveclin_task(stage: str) -> str:
    text = stage.casefold()
    if any(word in text for word in ("treat", "therap", "management", "intervention", "surg", "care", "resusc", "support")):
        return "treatment or management"
    if any(word in text for word in ("counsel", "prevention", "follow-up", "follow up", "surveillance", "prognos", "monitor")):
        return "follow-up, counseling, or prevention"
    if any(word in text for word in ("triage", "acuity", "disposition", "referral")):
        return "triage or referral"
    if any(word in text for word in ("diagnos", "imaging", "patholog", "histolog", "laboratory", "interpret", "assessment", "evaluation", "workup", "work-up", "testing", "findings")):
        return "diagnosis or diagnostic interpretation"
    return "clinical reasoning"


def _chapter(level: str) -> str:
    match = re.match(r"Chapter\s+(\d+)\s*:\s*(.+)", str(level).strip())
    if not match:
        return str(level).strip() or "Unmatched"
    number = int(match.group(1))
    return f"{ROMAN.get(number, str(number))}. {match.group(2).strip()}"


def _liveclin_id(pmc: object, mcq_number: int) -> str:
    digest = hashlib.sha256(f"LiveClin|2025_H1|{pmc}|{mcq_number}".encode()).hexdigest()[:16]
    return f"liveclin-{digest}"


def _fetch_liveclin() -> pd.DataFrame:
    cache = DATA / "reference" / f"liveclin-{LIVECLIN_REVISION[:12]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    response = requests.get(LIVECLIN_PARQUET, timeout=(30, 180))
    response.raise_for_status()
    frame = pd.read_parquet(io.BytesIO(response.content))
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def _base_public_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_parquet(DATA / "explorer_items.parquet")
    coded = pd.read_parquet(DATA / "explorer_coded.parquet")
    live_ids = set(coded.loc[coded["source"].eq("LiveClin"), "item_id"].astype(str))
    if live_ids:
        items = items[~items["item_id"].astype(str).isin(live_ids)].copy()
        coded = coded[~coded["item_id"].astype(str).isin(live_ids)].copy()
    source_by_id = coded.drop_duplicates("item_id").set_index("item_id")["source"].astype(str)
    items["source"] = items["item_id"].map(source_by_id).fillna("")
    return items, coded


def _harmonize_existing(items: pd.DataFrame, coded: pd.DataFrame, defaults: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for field in (
        "clinical_task", "care_setting", "user_role", "response_format",
        "evaluation_method", "reference_standard", "harmonization_provenance",
    ):
        items[field] = [
            _source_defaults(str(source), defaults)[field] for source in items["source"]
        ]
    items["native_stage"] = ""
    items["evaluation_availability"] = [
        "rubric_available_not_scored" if source == "HealthBench" else "answer_key_available_not_scored"
        for source in items["source"].astype(str)
    ]
    items["label_validated"] = False
    coded["code_system"] = coded["code"].apply(lambda value: "ICD-10-CM" if str(value) not in {"", "unmatched"} else "")
    coded["code_provenance"] = "AI-assigned condition; deterministic ICD-10-CM lookup"
    coded["label_provenance"] = "AI model"
    coded["label_scope"] = "question or answer option"
    coded["source_stage"] = ""
    return items, coded


def _liveclin_rows(frame: pd.DataFrame, defaults: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_rows: list[dict] = []
    coded_rows: list[dict] = []
    source_defaults = _source_defaults("LiveClin", defaults)
    for case in frame.to_dict("records"):
        policy = case["exam_creation"]["final_policy"]
        scenario = str(policy.get("scenario") or "").strip()
        code_field = str(case.get("ICD-10") or "").strip()
        code, _, condition = code_field.partition(" ")
        mcqs = policy.get("mcqs")
        if hasattr(mcqs, "tolist"):
            mcqs = mcqs.tolist()
        for number, mcq in enumerate(mcqs or []):
            stage = str(mcq.get("stage") or "").strip()
            question = str(mcq.get("question") or "").strip()
            options = dict(mcq.get("options") or {})
            answer_key = str(mcq.get("correct_answer") or "").strip()
            item_id = _liveclin_id(case.get("pmc"), number)
            full_question = f"{scenario}\n\n[{stage}] {question}" if scenario else f"[{stage}] {question}"
            item_rows.append({
                "item_id": item_id,
                "question": full_question,
                "question_redistribution_allowed": True,
                "license": "cc-by-nc-sa-4.0",
                "expected_answer_key": answer_key,
                "expected_answer": str(options.get(answer_key, answer_key)),
                "answer_options": json.dumps(options, ensure_ascii=False, sort_keys=True),
                "wrong_option_map": "{}",
                "source": "LiveClin",
                "clinical_task": _liveclin_task(stage),
                "care_setting": source_defaults["care_setting"],
                "user_role": source_defaults["user_role"],
                "response_format": source_defaults["response_format"],
                "evaluation_method": source_defaults["evaluation_method"],
                "reference_standard": source_defaults["reference_standard"],
                "harmonization_provenance": source_defaults["harmonization_provenance"],
                "native_stage": stage,
                "evaluation_availability": "answer_key_available_not_scored",
                "label_validated": False,
            })
            coded_rows.append({
                "item_id": item_id,
                "source": "LiveClin",
                "condition_name": condition or code_field,
                "role": "case_diagnosis",
                "no_condition_category": "",
                "code": code or "unmatched",
                "ccsr_category": "not_applicable",
                "ccsr_description": "Not mapped from ICD-10 source field",
                "icd_chapter": _chapter(case.get("Level1", "")),
                "code_system": "ICD-10",
                "code_provenance": "native case-level ICD-10 field",
                "label_provenance": "benchmark metadata",
                "label_scope": "case-level",
                "source_stage": stage,
            })
    return pd.DataFrame(item_rows), pd.DataFrame(coded_rows)


def _clean(values) -> list[str]:
    return sorted({str(value) for value in values if str(value) and str(value) not in {"unmatched", "not_applicable"}})


def _static_payload(items: pd.DataFrame, coded: pd.DataFrame, catalog: dict) -> dict:
    rows_by_item: dict[str, list[list[str]]] = {}
    for row in coded.itertuples(index=False):
        rows_by_item.setdefault(str(row.item_id), []).append([
            str(row.condition_name), str(row.role), str(row.no_condition_category),
            str(row.code), str(row.ccsr_description), str(row.icd_chapter),
            str(row.code_system), str(row.code_provenance), str(row.label_provenance),
            str(row.label_scope), str(row.source_stage),
        ])
    payload_items = []
    for row in items.itertuples(index=False):
        try:
            options = json.loads(str(row.answer_options or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            options = {}
        try:
            wrong = json.loads(str(row.wrong_option_map or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            wrong = {}
        allowed = bool(row.question_redistribution_allowed)
        answer = (
            f"{row.expected_answer_key} · {row.expected_answer}"
            if str(row.expected_answer_key) and str(row.expected_answer_key).casefold() != str(row.expected_answer).casefold()
            else str(row.expected_answer)
        )
        payload_items.append({
            "id": str(row.item_id), "s": str(row.source), "q": str(row.question), "a": answer,
            "k": str(row.expected_answer_key), "o": options, "w": wrong,
            "n": (
                f"Question shown; source license: {row.license}"
                if allowed else f"Question withheld; source license is {row.license or 'unknown'}"
            ),
            "t": str(row.clinical_task), "g": str(row.care_setting), "u": str(row.user_role),
            "f": str(row.response_format), "e": str(row.evaluation_method),
            "ref": str(row.reference_standard), "hp": str(row.harmonization_provenance),
            "stage": str(row.native_stage), "ev": str(row.evaluation_availability),
            "lv": bool(row.label_validated), "r": rows_by_item.get(str(row.item_id), []),
        })
    indexed = len(payload_items)
    validated = int(items["label_validated"].astype(bool).sum())
    return {
        "schema_version": "2.0",
        "generated_from": {
            "liveclin_revision": LIVECLIN_REVISION,
            "catalog_verified": catalog["last_verified"],
        },
        "denominators": {"n_indexed": indexed, "n_scored": 0, "n_label_validated": validated},
        "evaluation_note": "No reusable item-level model-result rows are attached yet. Coverage and evaluability are shown separately from performance.",
        "catalog": catalog["benchmarks"],
        "items": payload_items,
        "options": {
            "sources": _clean(coded["source"]),
            "categories": _clean(coded["ccsr_description"]),
            "chapters": _clean(coded["icd_chapter"]),
            "roles": ["tested", "context", "ruled_out", "wrong_option", "case_diagnosis"],
            "tasks": _clean(items["clinical_task"]),
            "settings": _clean(items["care_setting"]),
            "formats": _clean(items["response_format"]),
            "evaluation_states": _clean(items["evaluation_availability"]),
        },
    }


def _write_empty_evaluations() -> None:
    schema = _load_json(CONFIG / "evaluation_result_schema.json")
    frame = pd.DataFrame({name: pd.Series(dtype="object") for name in schema["fields"]})
    frame["score"] = pd.Series(dtype="float64")
    frame["correct"] = pd.Series(dtype="boolean")
    frame.to_parquet(DATA / "explorer_evaluations.parquet", index=False)


def main() -> None:
    defaults = _load_json(CONFIG / "source_harmonization.json")
    catalog = _load_json(CONFIG / "benchmark_catalog.json")
    items, coded = _base_public_data()
    items, coded = _harmonize_existing(items, coded, defaults)
    live_items, live_coded = _liveclin_rows(_fetch_liveclin(), defaults)
    items = pd.concat([items, live_items], ignore_index=True)
    coded = pd.concat([coded, live_coded], ignore_index=True)
    source_order = {
        name: index for index, name in enumerate([
            "MedQA", "MedMCQA", "PubMedQA", "MMLU anatomy", "MMLU clinical knowledge",
            "MMLU college medicine", "MMLU medical genetics", "MMLU professional medicine",
            "MedBullets", "HealthBench", "LiveClin",
        ])
    }
    items = items.drop_duplicates("item_id", keep="last")
    items["_source_order"] = items["source"].map(source_order).fillna(999)
    items = items.sort_values(["_source_order", "item_id"]).drop(columns="_source_order").reset_index(drop=True)
    coded = coded.drop_duplicates(
        ["item_id", "condition_name", "role", "code"], keep="last"
    )
    coded["_source_order"] = coded["source"].map(source_order).fillna(999)
    coded = coded.sort_values(
        ["_source_order", "item_id", "role", "condition_name"]
    ).drop(columns="_source_order").reset_index(drop=True)
    items.to_parquet(DATA / "explorer_items.parquet", index=False)
    coded.to_parquet(DATA / "explorer_coded.parquet", index=False)
    _write_empty_evaluations()
    payload = _static_payload(items, coded, catalog)
    STATIC_DATA.parent.mkdir(parents=True, exist_ok=True)
    STATIC_DATA.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        f"Wrote {len(items):,} indexed items, {len(coded):,} item-condition rows, "
        f"{len(catalog['benchmarks']):,} catalog entries, and 0 scored result rows."
    )


if __name__ == "__main__":
    main()
