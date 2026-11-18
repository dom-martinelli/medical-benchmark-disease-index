from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .collect import _redistribution_allowed
from .common import DATA, FIGURES, REPORT, append_log, ensure_dirs

ALL_CHAPTERS = [
    "I. Certain infectious and parasitic diseases", "II. Neoplasms", "III. Blood and immune disorders",
    "IV. Endocrine, nutritional and metabolic diseases", "V. Mental and behavioral disorders",
    "VI. Diseases of the nervous system", "VII. Diseases of the eye and adnexa",
    "VIII. Diseases of the ear and mastoid process", "IX. Diseases of the circulatory system",
    "X. Diseases of the respiratory system", "XI. Diseases of the digestive system",
    "XII. Diseases of the skin and subcutaneous tissue", "XIII. Musculoskeletal and connective tissue diseases",
    "XIV. Diseases of the genitourinary system", "XV. Pregnancy, childbirth and the puerperium",
    "XVI. Conditions originating in the perinatal period", "XVII. Congenital malformations and chromosomal abnormalities",
    "XVIII. Symptoms, signs and abnormal findings", "XIX. Injury, poisoning and certain other consequences",
    "XX. External causes of morbidity", "XXI. Factors influencing health status", "XXII. Codes for special purposes",
]


def _short_chapter(value: str) -> str:
    return value.split(". ", 1)[-1]


def expected_answer_text(options: object, answer_key: object) -> str:
    key = str(answer_key or "").strip()
    if not key:
        return ""
    try:
        parsed = json.loads(str(options or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {}
    if isinstance(parsed, dict):
        answer = parsed.get(key)
        if answer not in (None, ""):
            return str(answer)
        for value in parsed.values():
            if str(value).strip().casefold() == key.casefold():
                return str(value)
    return key


def wrong_option_choice_map(options: object, answer_key: object, labels: list[dict[str, object]]) -> dict[str, list[str]]:
    try:
        parsed = json.loads(str(options or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    expected = str(answer_key or "").strip().casefold()
    result: dict[str, list[str]] = {}
    for label in labels:
        condition = str(label.get("condition_name") or "").strip()
        quote = " ".join(str(label.get("supporting_quote") or "").casefold().split())
        condition_norm = " ".join(condition.casefold().split())
        if not condition_norm:
            continue
        normalized_options = {
            str(key): " ".join(str(value or "").casefold().split())
            for key, value in parsed.items()
        }
        expected_option = next(
            (value for key, value in normalized_options.items() if key.casefold() == expected), ""
        )
        if expected_option and condition_norm in expected_option:
            continue
        incorrect = {key: value for key, value in normalized_options.items() if key.casefold() != expected}
        matches = [key for key, value in incorrect.items() if condition_norm in value]
        if not matches:
            quote_matches = [key for key, value in incorrect.items() if len(value) >= 4 and value in quote]
            matches = quote_matches if len(quote_matches) == 1 else []
        for key_text in matches:
            result.setdefault(key_text, [])
            if condition not in result[key_text]:
                result[key_text].append(condition)
    return result


def _figures(coded: pd.DataFrame) -> None:
    tested = coded[(coded["role"] == "tested") & (coded["code"] != "unmatched")].copy()
    chapter = tested.groupby(["icd_chapter", "source"])["item_id"].nunique().unstack(fill_value=0)
    chapter = chapter.reindex([x for x in ALL_CHAPTERS if x in chapter.index])
    if not chapter.empty:
        ax = chapter.plot(kind="barh", stacked=True, figsize=(12, 10), colormap="tab20")
        ax.set_xlabel("Distinct benchmark items")
        ax.set_ylabel("ICD-10-CM chapter")
        ax.set_yticklabels([_short_chapter(x) for x in chapter.index], fontsize=8)
        ax.set_title("Tested-condition coverage by ICD chapter and benchmark")
        ax.legend(title="Benchmark", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES / "icd_chapter_by_benchmark.png", dpi=180, bbox_inches="tight")
        plt.close()

    distribution = tested.groupby("ccsr_description")["item_id"].nunique().sort_values(ascending=False).head(20).sort_values()
    if not distribution.empty:
        plt.figure(figsize=(11, 8))
        plt.barh(distribution.index, distribution.values, color="#176b87")
        plt.xlabel("Distinct benchmark items")
        plt.ylabel("CCSR category")
        plt.title("Twenty most represented tested-condition categories")
        plt.tight_layout()
        plt.savefig(FIGURES / "disease_distribution.png", dpi=180, bbox_inches="tight")
        plt.close()


def prepare_explorer_data(items: pd.DataFrame, coded: pd.DataFrame) -> None:
    item_columns = [
        "item_id", "question", "options", "answer_key", "has_answer_key",
        "question_redistribution_allowed", "license",
    ]
    explorer_items = items[item_columns].copy()
    # Recompute from the current rule rather than trusting the stored column,
    # which predates the HealthBench no-exposure request in older collections.
    explorer_items["question_redistribution_allowed"] = [
        _redistribution_allowed(str(lic or ""), str(src)) for lic, src in zip(items["license"], items["source"])
    ]
    explorer_items.loc[~explorer_items["question_redistribution_allowed"], "question"] = (
        "[Question text withheld because redistribution permission is unclear.]"
    )
    explorer_items["expected_answer_key"] = explorer_items["answer_key"].fillna("").astype(str)
    explorer_items["expected_answer"] = [
        expected_answer_text(options, key)
        for options, key in zip(explorer_items["options"], explorer_items["answer_key"])
    ]
    no_key = ~explorer_items["has_answer_key"].astype(bool)
    withheld = (~explorer_items["question_redistribution_allowed"]) & (~no_key)
    explorer_items.loc[no_key, ["expected_answer_key", "expected_answer"]] = [
        "", "[No public expected answer provided.]"
    ]
    explorer_items.loc[withheld, ["expected_answer_key", "expected_answer"]] = [
        "", "[Expected answer withheld because redistribution permission is unclear.]"
    ]
    wrong_rows = coded[coded["role"].eq("wrong_option")][
        ["item_id", "condition_name", "supporting_quote"]
    ]
    wrong_by_item = {
        str(item_id): group[["condition_name", "supporting_quote"]].to_dict("records")
        for item_id, group in wrong_rows.groupby("item_id", sort=False)
    }
    explorer_items["answer_options"] = explorer_items["options"].fillna("{}").astype(str)
    explorer_items["wrong_option_map"] = [
        json.dumps(
            wrong_option_choice_map(options, key, wrong_by_item.get(str(item_id), [])),
            ensure_ascii=False,
        )
        for item_id, options, key in zip(
            explorer_items["item_id"], explorer_items["options"], explorer_items["answer_key"]
        )
    ]
    explorer_items.loc[~explorer_items["question_redistribution_allowed"], "answer_options"] = "{}"
    explorer_items.loc[~explorer_items["question_redistribution_allowed"], "wrong_option_map"] = "{}"
    explorer_items = explorer_items.drop(columns=["options", "answer_key", "has_answer_key"])
    explorer_items.to_parquet(DATA / "explorer_items.parquet", index=False)
    coded_columns = [
        "item_id", "source", "condition_name", "role", "no_condition_category", "code",
        "ccsr_category", "ccsr_description", "icd_chapter",
    ]
    coded[coded_columns].to_parquet(DATA / "explorer_coded.parquet", index=False)

def run() -> str:
    ensure_dirs()
    required = [DATA / "items.parquet", DATA / "coded.parquet", DATA / "validation.parquet"]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 5 requires completed earlier stages; missing: {', '.join(missing)}")
    items = pd.read_parquet(DATA / "items.parquet")
    coded = pd.read_parquet(DATA / "coded.parquet")
    validation = pd.read_parquet(DATA / "validation.parquet")
    _figures(coded)
    prepare_explorer_data(items, coded)

    source_counts = items.groupby("source").agg(
        items=("item_id", "nunique"),
        with_answer_key=("has_answer_key", "sum"),
        without_answer_key=("has_answer_key", lambda x: int((~x).sum())),
    ).reset_index()
    tested = coded[coded["role"] == "tested"]
    distribution = (
        tested[tested["code"] != "unmatched"]
        .groupby(["ccsr_category", "ccsr_description"])["item_id"].nunique()
        .sort_values(ascending=False).head(20).rename("items").reset_index()
    )
    chapter_counts = tested[tested["code"] != "unmatched"].groupby("icd_chapter")["item_id"].nunique()
    missing_chapters = [x for x in ALL_CHAPTERS if int(chapter_counts.get(x, 0)) == 0]
    sparse_chapters = [(x, int(chapter_counts.get(x, 0))) for x in ALL_CHAPTERS if 0 < int(chapter_counts.get(x, 0)) < 10]
    failed_items = int(coded.loc[coded["no_condition_category"] == "label_failed", "item_id"].nunique())
    unmatched = int(((coded["condition_name"] != "") & (coded["code"] == "unmatched")).sum())
    condition_rows = int((coded["condition_name"] != "").sum())
    v = validation.iloc[0]
    kappa = float(v["cohens_kappa_overall"])
    exact = float(v["exact_item_agreement_overall"])
    repeat = float(v["repeat_agreement_overall"])
    unsupported = float(v["unsupported_share_overall"])
    secondary_attempted = int(v.get("secondary_attempted_overall", len(validation)))
    secondary_failed = int(v.get("secondary_failed_overall", secondary_attempted - len(validation)))
    repeat_attempted = int(v.get("repeat_attempted_overall", 100))
    repeat_failed = int(v.get("repeat_failed_overall", 0))
    repeat_scored = int(v.get("repeat_scored_overall", repeat_attempted - repeat_failed))
    review_attempted = int(v.get("review_attempted_overall", len(validation)))
    review_failed = int(v.get("review_failed_overall", 0))
    reviewed_labels = int(v.get("reviewed_labels_overall", 0))
    unsupported_labels = int(v.get("unsupported_labels_overall", round(unsupported * reviewed_labels)))

    missing_text = ", ".join(_short_chapter(x) for x in missing_chapters) or "none"
    sparse_text = ", ".join(f"{_short_chapter(x)} ({n})" for x, n in sparse_chapters) or "none"
    report = f"""# Disease coverage in public medical AI benchmarks: draft methods and results

## Overview

This project assembled a question-level index of public medical AI benchmarks and labeled the clinical conditions involved in each item. The resulting index is intended to make benchmark coverage inspectable by disease rather than only by benchmark name or broad academic subject. The analysis collected {len(items):,} unique items. Of these, {int(items['has_answer_key'].sum()):,} had a public answer key and {int((~items['has_answer_key']).sum()):,} did not. Items without a key were retained because they still describe what a benchmark asks a model to address, but they were not treated as ordinary scored multiple-choice questions.

## Methods

The collection stage retrieved test and validation data for MedQA and five medical MMLU subjects, the MedMCQA validation split, MedBullets test items, and HealthBench test prompts. PubMedQA's labeled subset exposes only a train split, so that split was included as a documented exception. The two available MedBullets files are derived four- and five-option versions of the same parent questions; the five-option record was retained once and its explanation was attached to that row. Cross-source near duplicates were flagged with TF-IDF cosine similarity at a threshold of 0.94 and were not removed.

Every item was passed to the selected language model with its question, options, answer key, and explanation. The model returned condition names in plain clinical language, a role for each condition, a short supporting quotation, and a high or low confidence value. Roles distinguished the condition being tested from context, explicit exclusions, and diseases appearing only in wrong answer options. Items with no condition were assigned to a non-disease category. The labeling prompt, model, settings, and run date were stored with every output row.

Condition names were mapped without asking a language model to recall codes. The pipeline matched names to the pinned ICD-10-CM description file by exact text, fuzzy similarity, and then sentence-embedding similarity. Matches were accepted only above the configured thresholds. Codes were rolled up with AHRQ CCSR v2026.1; conditions below threshold remained explicitly unmatched. This design avoids silently inventing an ICD code.

Validation used three automatic checks. A prespecified random sample of {secondary_attempted:,} items was submitted to a second model and compared at the tested-condition CCSR-category level; {len(validation):,} produced valid outputs and {secondary_failed:,} failed after retries. A {repeat_attempted}-item subset was labeled again by the primary model, with {repeat_failed:,} failed outputs. A third model reviewed whether each original condition and role was supported by the item; valid reviewer output was obtained for {review_attempted - review_failed:,} of {review_attempted:,} items. The associated disagreement page presents no more than 60 of the worst cases with both label sets side by side. Naveed or Carmen could review that focused page in under an hour.

## Collection results

{source_counts.to_markdown(index=False)}

## Disease coverage

The most represented tested-condition CCSR categories are shown below. Counts are distinct items, so an item with repeated mentions of the same category contributes once within a category.

{distribution.to_markdown(index=False)}

Across the ICD-10-CM chapters, chapters with no tested-condition items were: {missing_text}. Chapters with fewer than ten items were: {sparse_text}. Figure 1 shows chapter coverage by benchmark, and Figure 2 shows the twenty most represented CCSR categories.

![Tested-condition coverage by ICD chapter and benchmark](figures/icd_chapter_by_benchmark.png)

![Twenty most represented tested-condition categories](figures/disease_distribution.png)

## Label validation

Among the {len(validation):,} items with valid second-model output, the two models assigned exactly the same set of tested-condition CCSR categories on {exact:.1%}. Cohen's kappa across item-by-category decisions was {kappa:.3f}. Across {repeat_scored:,} valid repeat outputs, the primary model's exact condition-and-role output agreed {repeat:.1%} of the time. The reviewer model marked {unsupported_labels:,} of {reviewed_labels:,} successfully reviewed proposed labels ({unsupported:.1%}) as unsupported. Reviewer output failed validation for {review_failed:,} of {review_attempted:,} sampled items, so this support estimate applies only to successful reviewer outputs. These numbers describe reproducibility and model-based review, not clinical ground truth.

## Limitations

The index cannot recover answer keys that benchmark publishers did not release. It therefore preserves {int((~items['has_answer_key']).sum()):,} such items but does not pretend their correct answer is known. The mapping stage left {unmatched:,} of {condition_rows:,} condition rows unmatched ({unmatched / condition_rows if condition_rows else 0:.1%}); these conditions remain in the data and can be audited rather than disappearing from coverage estimates. {failed_items:,} items could not be labeled after repeated attempts; they are kept in the index, flagged `label_failed`, and excluded from disease counts only. Fuzzy and embedding matches are reproducible lookup results, but they can still select a clinically broader or narrower code than a human coder would choose.

Automatic agreement does not establish validity because language models can share errors or training data. The clinician disagreement review is therefore the most useful next quality check. Benchmark composition also reflects what dataset creators chose to publish, the availability of public keys and licenses, and the conventions of ICD-10-CM and CCSR. The explorer suppresses question text from sources whose current dataset cards do not clearly permit redistribution; it shows item identifiers and derived labels instead.

## Data availability and reproducibility

Each stage writes a durable output and a resumable checkpoint. `data/items.parquet`, `data/labels.parquet`, `data/coded.parquet`, and `data/validation.parquet` contain the successive analysis layers. `run_log.md` records prompt snapshots, model choices, settings, counts, tokens, cost, and timing. Official coding files are pinned by version and SHA-256 in `data/reference/manifest.json`.
"""
    path = REPORT / "draft_report.md"
    path.write_text(report, encoding="utf-8")
    append_log("Stage 5 report", f"Draft report written: `{path.relative_to(path.parents[1])}`  \nFigures written: {len(list(FIGURES.glob('*.png')))}  \nRedacted explorer data written: `data/explorer_items.parquet`, `data/explorer_coded.parquet`")
    print(f"Stage 5 complete: wrote {path} and {len(list(FIGURES.glob('*.png')))} PNG figures.")
    return report
