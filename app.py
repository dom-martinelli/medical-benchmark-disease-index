from __future__ import annotations

import json
import tempfile
import os
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
DEFAULT_CODED = ROOT / "data" / "explorer_coded.parquet"
DEFAULT_ITEMS = ROOT / "data" / "explorer_items.parquet"
DATA_PATH = Path(os.getenv("BENCHMARK_CODED_PATH", DEFAULT_CODED if DEFAULT_CODED.exists() else ROOT / "data" / "coded.parquet"))
ITEMS_PATH = Path(os.getenv("BENCHMARK_ITEMS_PATH", DEFAULT_ITEMS if DEFAULT_ITEMS.exists() else ROOT / "data" / "items.parquet"))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists() or not ITEMS_PATH.exists():
        raise FileNotFoundError("Run the pipeline through Stage 3 before starting the explorer.")
    coded = pd.read_parquet(DATA_PATH)
    items = pd.read_parquet(ITEMS_PATH)
    return coded, items


CODED, ITEMS = load_data()


def _choices(column: str) -> list[str]:
    return sorted(x for x in CODED[column].dropna().astype(str).unique() if x and x != "unmatched")


def _item_choices(column: str) -> list[str]:
    return sorted(x for x in ITEMS[column].dropna().astype(str).unique() if x)


def filter_rows(
    query: str,
    sources: list[str],
    categories: list[str],
    chapters: list[str],
    roles: list[str],
    tasks: list[str],
    settings: list[str],
    response_formats: list[str],
    evaluation_states: list[str],
):
    rows = CODED.copy()
    if query and query.strip():
        needle = query.strip().casefold()
        rows = rows[
            rows["condition_name"].fillna("").str.casefold().str.contains(needle, regex=False)
            | rows["code"].fillna("").str.casefold().str.contains(needle, regex=False)
        ]
    if sources:
        rows = rows[rows["source"].isin(sources)]
    if categories:
        rows = rows[rows["ccsr_category"].isin(categories) | rows["ccsr_description"].isin(categories)]
    if chapters:
        rows = rows[rows["icd_chapter"].isin(chapters)]
    if roles:
        rows = rows[rows["role"].isin(roles)]
    item_filter = ITEMS
    if tasks:
        item_filter = item_filter[item_filter["clinical_task"].isin(tasks)]
    if settings:
        item_filter = item_filter[item_filter["care_setting"].isin(settings)]
    if response_formats:
        item_filter = item_filter[item_filter["response_format"].isin(response_formats)]
    if evaluation_states:
        item_filter = item_filter[item_filter["evaluation_availability"].isin(evaluation_states)]
    rows = rows[rows["item_id"].isin(item_filter["item_id"])]

    merged = rows.merge(
        ITEMS[[
            "item_id", "question", "expected_answer_key", "expected_answer",
            "answer_options", "wrong_option_map", "question_redistribution_allowed", "license",
            "clinical_task", "care_setting", "user_role", "response_format", "native_stage",
            "evaluation_method", "reference_standard", "evaluation_availability",
            "harmonization_provenance",
        ]],
        on="item_id", how="left", suffixes=("", "_item"),
    )
    if merged.empty:
        table = pd.DataFrame(columns=[
            "item_id", "source", "question", "expected_answer_key", "expected_answer",
            "answer_options", "answer_options_plain",
            "clinical_task", "care_setting", "user_role", "response_format", "native_stage",
            "conditions", "codes", "code_provenance", "evaluation_method",
            "reference_standard", "evaluation_availability", "license_note"
        ])
        chart = px.bar(title="No matching items")
    else:
        merged["condition_display"] = merged.apply(
            lambda x: f"{x['condition_name']} [{x['role']}]" if x["condition_name"] else f"[{x['no_condition_category']}]", axis=1
        )
        merged["code_display"] = merged.apply(
            lambda x: f"{x['code']} · {x['ccsr_description']}" if x["code"] != "unmatched" else "unmatched", axis=1
        )
        grouped = merged.groupby(["item_id", "source"], sort=False, as_index=False).agg(
            question=("question", "first"),
            expected_answer_key=("expected_answer_key", "first"),
            expected_answer=("expected_answer", "first"),
            answer_options=("answer_options", "first"),
            wrong_option_map=("wrong_option_map", "first"),
            allowed=("question_redistribution_allowed", "first"),
            license=("license", "first"),
            clinical_task=("clinical_task", "first"),
            care_setting=("care_setting", "first"),
            user_role=("user_role", "first"),
            response_format=("response_format", "first"),
            native_stage=("native_stage", "first"),
            evaluation_method=("evaluation_method", "first"),
            reference_standard=("reference_standard", "first"),
            evaluation_availability=("evaluation_availability", "first"),
            conditions=("condition_display", lambda x: "; ".join(dict.fromkeys(x))),
            codes=("code_display", lambda x: "; ".join(dict.fromkeys(x))),
            code_provenance=("code_provenance", lambda x: "; ".join(dict.fromkeys(x))),
        )
        grouped["license_note"] = grouped.apply(
            lambda x: f"Question shown; source card license: {x['license']}" if x["allowed"]
            else f"Question withheld; source card license is {x['license'] or 'unknown'}", axis=1,
        )
        grouped.loc[~grouped["allowed"], "question"] = "[Question text withheld because redistribution permission is unclear.]"
        grouped["expected_answer"] = grouped.apply(
            lambda x: f"{x['expected_answer_key']} · {x['expected_answer']}" if x["expected_answer_key"]
            and x["expected_answer_key"].casefold() != x["expected_answer"].casefold() else x["expected_answer"],
            axis=1,
        )
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
            # Same options with no [expected]/[incorrect] markers. This is the
            # column to feed a model: answer_options labels the correct choice
            # in plain text, so prompting with it scores near 100% by leakage.
            try:
                options = json.loads(row["answer_options"] or "{}")
            except (TypeError, ValueError):
                return ""
            if not options:
                return ""
            return "\n".join(f"{key}. {value}" for key, value in options.items())

        grouped["answer_options_plain"] = grouped.apply(plain_options, axis=1)
        grouped["answer_options"] = grouped.apply(display_options, axis=1)
        table = grouped[[
            "item_id", "source", "question", "expected_answer_key", "expected_answer",
            "answer_options", "answer_options_plain",
            "clinical_task", "care_setting", "user_role", "response_format", "native_stage",
            "conditions", "codes", "code_provenance", "evaluation_method",
            "reference_standard", "evaluation_availability", "license_note"
        ]]
        counts = (
            merged[merged["icd_chapter"] != "Unmatched"]
            .groupby(["icd_chapter", "source"])["item_id"].nunique().rename("items").reset_index()
        )
        chart = px.bar(
            counts, x="items", y="icd_chapter", color="source", orientation="h", barmode="stack",
            labels={"items": "Distinct items", "icd_chapter": "ICD-10-CM chapter", "source": "Benchmark"},
            title="Filtered item counts per ICD chapter and benchmark",
        )
        chart.update_layout(height=max(420, 28 * counts["icd_chapter"].nunique()), yaxis={"categoryorder": "total ascending"})
    handle = tempfile.NamedTemporaryFile(prefix="benchmark-index-", suffix=".csv", delete=False)
    handle.close()
    table.to_csv(handle.name, index=False)
    summary = (
        f"n_indexed = {len(table):,} · n_scored = 0 · n_label_validated = 0 · "
        f"{len(rows):,} matching item-condition rows"
    )
    return table, chart, handle.name, summary


CSS = """
.gradio-container {max-width: 1500px !important;}
#title {background: linear-gradient(120deg,#133f4c,#176b87);color:white;padding:28px;border-radius:16px;}
#title h1 {margin:0;font-size:2rem;} #summary {font-weight:650;color:#176b87;}
"""

with gr.Blocks(title="Medical benchmark disease index") as demo:
    gr.Markdown(
        "# Medical benchmark disease index\nSearch what public medical AI benchmarks actually cover by condition and standard disease code.",
        elem_id="title",
    )
    with gr.Row():
        query = gr.Textbox(label="Condition or ICD-10-CM code", placeholder="e.g., diabetes or E11")
        source = gr.Dropdown(_choices("source"), multiselect=True, label="Benchmark")
    with gr.Row():
        category = gr.Dropdown(_choices("ccsr_description"), multiselect=True, label="CCSR category")
        chapter = gr.Dropdown(_choices("icd_chapter"), multiselect=True, label="ICD chapter")
        role = gr.Dropdown(["tested", "context", "ruled_out", "wrong_option", "case_diagnosis"], multiselect=True, label="Condition role")
    with gr.Row():
        task = gr.Dropdown(_item_choices("clinical_task"), multiselect=True, label="Clinical task")
        setting = gr.Dropdown(_item_choices("care_setting"), multiselect=True, label="Care setting")
        response_format = gr.Dropdown(_item_choices("response_format"), multiselect=True, label="Response format")
        evaluation_state = gr.Dropdown(_item_choices("evaluation_availability"), multiselect=True, label="Evaluation availability")
    with gr.Row():
        apply_button = gr.Button("Apply filters", variant="primary")
        clear_button = gr.ClearButton([query, source, category, chapter, role, task, setting, response_format, evaluation_state], value="Clear filters")
        download = gr.DownloadButton("Download filtered CSV")
    summary = gr.Markdown(elem_id="summary")
    results = gr.Dataframe(label="Results", interactive=False, wrap=True, max_height=650)
    chart = gr.Plot(label="Coverage chart")
    inputs = [query, source, category, chapter, role, task, setting, response_format, evaluation_state]
    outputs = [results, chart, download, summary]
    apply_button.click(filter_rows, inputs=inputs, outputs=outputs)
    query.submit(filter_rows, inputs=inputs, outputs=outputs)
    demo.load(lambda: filter_rows("", [], [], [], [], [], [], [], []), outputs=outputs)

if __name__ == "__main__":
    demo.launch(css=CSS)
