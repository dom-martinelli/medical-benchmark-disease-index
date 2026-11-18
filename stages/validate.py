from __future__ import annotations

import html
import json
import math
import random
from datetime import date
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .code import CodeLookup, load_reference
from .common import CHECKPOINTS, DATA, PROMPTS, REVIEW, append_jsonl, append_log, json_for_prompt, log_prompt, prompt_version, read_jsonl, write_parquet_atomic
from .label import label_one
from .models import ModelClient, choose_distinct_local_model, choose_primary_model

SECONDARY_CP = CHECKPOINTS / "validation_secondary.jsonl"
REPEAT_CP = CHECKPOINTS / "validation_repeat.jsonl"
REVIEW_CP = CHECKPOINTS / "validation_reviews.jsonl"


def _primary_model_name(coded: pd.DataFrame) -> str:
    value = str(coded.iloc[0]["model_name"])
    return value.split("/", 1)[-1]


def _group_labels(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["item_id"]), []).append(row)
    return grouped


def _run_label_sample(
    sample: pd.DataFrame,
    client: ModelClient,
    checkpoint,
    prompt_text: str,
    label_kind: str,
) -> dict[str, list[dict[str, Any]]]:
    model_full = f"{client.provider}/{client.model_name}"
    settings_text = json.dumps(client.settings, sort_keys=True)
    version = prompt_version(PROMPTS / "label_conditions.md")
    valid_ids = set(sample["item_id"].astype(str))
    matching = [
        row for row in read_jsonl(checkpoint)
        if str(row.get("item_id")) in valid_ids
        and row.get("model_name") == model_full
        and row.get("model_settings") == settings_text
        and row.get("prompt_version") == version
        and row.get("validation_kind") == label_kind
    ]
    existing = _group_labels(matching)
    for number, item in enumerate(sample.to_dict("records"), start=1):
        if item["item_id"] in existing:
            continue
        validation_error = None
        try:
            labels = label_one(item, client, prompt_text)
        except Exception as exc:
            # Validation should remain auditable and resumable even when a local
            # model cannot produce valid JSON for one unusually long item. Keep
            # the failure in the checkpoint and exclude it from the denominator
            # rather than aborting the entire prespecified sample.
            labels = [{
                "condition_name": "",
                "role": "validation_failed",
                "supporting_quote": "",
                "confidence": "low",
                "no_condition_category": "validation_failed",
            }]
            validation_error = str(exc)
        output = [
            {
                "item_id": item["item_id"], "source": item["source"], **label,
                "model_name": model_full,
                "model_settings": settings_text,
                "prompt_text": prompt_text, "prompt_version": version,
                "run_date": date.today().isoformat(), "validation_kind": label_kind,
                "validation_failed": validation_error is not None,
                "validation_error": validation_error,
            }
            for label in labels
        ]
        append_jsonl(checkpoint, output)
        existing[item["item_id"]] = output
        if number % 100 == 0:
            print(f"Validation {label_kind}: processed {number:,} of {len(sample):,} items.")
    return existing


def _condition_set(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(x.get("condition_name", "")).casefold().strip(), str(x.get("role", "")).casefold().strip())
        for x in rows if str(x.get("condition_name", "")).strip()
    }


def _category_set(rows: list[dict[str, Any]], lookup: CodeLookup) -> set[str]:
    output = set()
    for row in rows:
        if row.get("role") != "tested" or not row.get("condition_name"):
            continue
        mapping = lookup.match(str(row["condition_name"]))
        if mapping["ccsr_category"] != "unmatched":
            output.add(mapping["ccsr_category"])
    return output


def _review_sample(
    sample: pd.DataFrame,
    primary: dict[str, list[dict[str, Any]]],
    reviewer: ModelClient,
    prompt_text: str,
) -> dict[str, dict[str, Any]]:
    model_full = f"{reviewer.provider}/{reviewer.model_name}"
    settings_text = json.dumps(reviewer.settings, sort_keys=True)
    version = prompt_version(PROMPTS / "review_labels.md")
    valid_ids = set(sample["item_id"].astype(str))
    existing = {
        str(x["item_id"]): x for x in read_jsonl(REVIEW_CP)
        if str(x.get("item_id")) in valid_ids
        and x.get("model_name") == model_full
        and x.get("model_settings") == settings_text
        and x.get("prompt_version") == version
    }
    for number, item in enumerate(sample.to_dict("records"), start=1):
        item_id = item["item_id"]
        labels = [
            {
                "label_index": index,
                "condition_name": row.get("condition_name"),
                "role": row.get("role"),
                "supporting_quote": row.get("supporting_quote"),
            }
            for index, row in enumerate(
                x for x in primary.get(item_id, []) if x.get("condition_name")
            )
        ]
        prior = existing.get(item_id)
        # A reviewer call is unnecessary when the primary model proposed no
        # conditions. Older checkpoints may contain an extraneous model review
        # for those empty cases; supersede it with the deterministic empty result.
        if prior and not (prior.get("validation_failed") and not labels):
            continue
        payload = {"item": json.loads(json_for_prompt(item)), "labels": labels}
        rendered = prompt_text.replace("{{ITEM_AND_LABELS_JSON}}", json.dumps(payload, ensure_ascii=False))
        last_error = None
        if not labels:
            reviews = []
        else:
            for attempt in range(3):
                try:
                    result = reviewer.generate_json(rendered)
                    reviews = result.get("reviews") if isinstance(result.get("reviews"), list) else []
                    indices = {int(x.get("label_index", -1)) for x in reviews if isinstance(x, dict)}
                    if indices != set(range(len(labels))):
                        raise ValueError(f"Reviewer returned indices {sorted(indices)} for {len(labels)} labels")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            else:
                reviews = []
        row = {
            "item_id": item_id,
            "reviews": reviews,
            "model_name": model_full,
            "model_settings": settings_text,
            "prompt_text": prompt_text,
            "prompt_version": version,
            "run_date": date.today().isoformat(),
            "validation_failed": last_error is not None and not reviews,
            "validation_error": str(last_error) if last_error is not None and not reviews else None,
        }
        append_jsonl(REVIEW_CP, [row])
        existing[item_id] = row
        if number % 100 == 0:
            print(f"Validation reviewer: processed {number:,} of {len(sample):,} items.")
    return existing


def _disagreement_html(rows: pd.DataFrame, items: pd.DataFrame) -> None:
    # ``rows`` already carries the sampled item's source. Merging a second
    # source column would rename both to source_x/source_y and break rendering.
    merged = rows.merge(items[["item_id", "question", "options", "answer_key"]], on="item_id", how="left")
    merged = merged[merged["disagreement_score"] > 0].sort_values("disagreement_score", ascending=False).head(60)
    cards = []
    for row in merged.to_dict("records"):
        cards.append(
            f"""<article class="card">
<div class="meta"><strong>{html.escape(str(row['source']))}</strong> · {html.escape(str(row['item_id']))} · score {row['disagreement_score']:.2f}</div>
<h2>{html.escape(str(row['question']))}</h2>
<pre>{html.escape(str(row['options']))}</pre>
<div class="grid"><section><h3>Primary labels</h3><pre>{html.escape(str(row['primary_labels']))}</pre></section>
<section><h3>Second-model labels</h3><pre>{html.escape(str(row['secondary_labels']))}</pre></section></div>
<p><strong>Repeat identical:</strong> {row['repeat_same']} &nbsp; <strong>Unsupported share:</strong> {row['unsupported_share']:.1%}</p>
</article>"""
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Benchmark label disagreements</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#f4f1ea;color:#18211d;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 20px}}
h1{{font-size:2.3rem;margin-bottom:.25rem}}.lede{{color:#506058;margin-bottom:2rem}}.card{{background:white;border:1px solid #d8d4ca;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 4px 20px #0000000c}}
.meta{{color:#526159;font-size:.9rem}}h2{{font-size:1.1rem}}h3{{font-size:.95rem}}pre{{white-space:pre-wrap;background:#f7f7f3;padding:12px;border-radius:8px;overflow:auto}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Labels needing review</h1><p class="lede">Worst automatic disagreements first. A clinician can review this 30–60-item page in under an hour.</p>{''.join(cards) or '<p>No disagreements were found.</p>'}</main></body></html>"""
    (REVIEW / "disagreements.html").write_text(document, encoding="utf-8")


def run(allow_paid: bool = False, allow_pull: bool = True) -> pd.DataFrame:
    if not (DATA / "coded.parquet").exists():
        raise FileNotFoundError("Stage 3 output data/coded.parquet is missing.")
    items = pd.read_parquet(DATA / "items.parquet")
    coded = pd.read_parquet(DATA / "coded.parquet")
    primary = {item_id: frame.to_dict("records") for item_id, frame in coded.groupby("item_id", sort=False)}
    label_prompt = (PROMPTS / "label_conditions.md").read_text(encoding="utf-8")
    reviewer_prompt = (PROMPTS / "review_labels.md").read_text(encoding="utf-8")
    log_prompt(PROMPTS / "label_conditions.md", "Stage 4 relabel")
    log_prompt(PROMPTS / "review_labels.md", "Stage 4 review")

    primary_client, _ = choose_primary_model(allow_paid=allow_paid, allow_pull=allow_pull)
    primary_full = str(coded.iloc[0]["model_name"])
    primary_provider, primary_name = primary_full.split("/", 1)
    if (primary_client.provider, primary_client.model_name) != (primary_provider, primary_name):
        raise RuntimeError(
            f"The original Stage 2 model {primary_full} is not the model now available for the repeat check "
            f"({primary_client.provider}/{primary_client.model_name}). Restore the original model and rerun Stage 4."
        )
    second = choose_distinct_local_model(
        {primary_name}, ["llama3.1:8b", "gemma3:4b", "llama3.2:3b"], allow_pull=allow_pull
    )
    reviewer = choose_distinct_local_model(
        {primary_name, second.model_name}, ["llama3.1:8b", "gemma3:4b", "llama3.2:3b", "qwen2.5:7b-instruct"], allow_pull=allow_pull
    )
    append_log("Stage 4 model choices", f"Primary/repeat: `{primary_client.model_name}`  \nAgreement model: `{second.model_name}`  \nReviewer model: `{reviewer.model_name}`")

    rng = random.Random(20260911)
    ids = items["item_id"].tolist()
    agreement_order = rng.sample(ids, min(300, len(ids)))
    agreement_ids = set(agreement_order)
    repeat_ids = set(rng.sample(agreement_order, min(100, len(agreement_order))))
    agreement_sample = items[items["item_id"].isin(agreement_ids)]
    repeat_sample = items[items["item_id"].isin(repeat_ids)]
    secondary = _run_label_sample(agreement_sample, second, SECONDARY_CP, label_prompt, "second_model")
    repeated = _run_label_sample(repeat_sample, primary_client, REPEAT_CP, label_prompt, "repeat_run")
    reviews = _review_sample(agreement_sample, primary, reviewer, reviewer_prompt)

    secondary_failed = {
        item_id for item_id, values in secondary.items()
        if any(bool(row.get("validation_failed")) for row in values)
    }
    repeat_failed = {
        item_id for item_id, values in repeated.items()
        if any(bool(row.get("validation_failed")) for row in values)
    }
    review_failed = {
        item_id for item_id, value in reviews.items()
        if bool(value.get("validation_failed"))
    }
    successful_reviews = [
        value for item_id, value in reviews.items() if item_id not in review_failed
    ]
    reviewed_labels = [
        review for value in successful_reviews for review in value.get("reviews", [])
        if isinstance(review, dict)
    ]
    unsupported_labels = sum(not bool(review.get("supported")) for review in reviewed_labels)
    repeat_results = {
        item_id: _condition_set(primary.get(item_id, [])) == _condition_set(repeated.get(item_id, []))
        for item_id in repeat_ids if item_id not in repeat_failed
    }

    reference, _ = load_reference()
    lookup = CodeLookup(reference)
    rows = []
    primary_sets: dict[str, set[str]] = {}
    secondary_sets: dict[str, set[str]] = {}
    for item in agreement_sample.to_dict("records"):
        item_id = item["item_id"]
        if item_id in secondary_failed:
            continue
        p_rows = primary.get(item_id, [])
        s_rows = secondary.get(item_id, [])
        p_set = {str(x["ccsr_category"]) for x in p_rows if x.get("role") == "tested" and x.get("ccsr_category") != "unmatched"}
        s_set = _category_set(s_rows, lookup)
        primary_sets[item_id], secondary_sets[item_id] = p_set, s_set
        review_values = reviews.get(item_id, {}).get("reviews", [])
        supported = [bool(x.get("supported")) for x in review_values if isinstance(x, dict)]
        unsupported_share = (
            float("nan") if item_id in review_failed
            else 1 - sum(supported) / len(supported) if supported else 0.0
        )
        repeat_same = repeat_results.get(item_id)
        category_agreement = p_set == s_set
        score = (
            (0 if category_agreement else 1)
            + (0 if repeat_same is None or repeat_same else 1)
            + (0 if math.isnan(unsupported_share) else unsupported_share)
        )
        rows.append(
            {
                "item_id": item_id, "source": item["source"],
                "primary_categories": json.dumps(sorted(p_set)), "secondary_categories": json.dumps(sorted(s_set)),
                "category_agreement": category_agreement, "repeat_same": repeat_same,
                "unsupported_share": unsupported_share, "disagreement_score": score,
                "repeat_failed": item_id in repeat_failed,
                "review_failed": item_id in review_failed,
                "primary_labels": json.dumps([(x.get("condition_name"), x.get("role")) for x in p_rows], ensure_ascii=False),
                "secondary_labels": json.dumps([(x.get("condition_name"), x.get("role")) for x in s_rows], ensure_ascii=False),
                "primary_model": str(coded.iloc[0]["model_name"]), "second_model": f"{second.provider}/{second.model_name}",
                "reviewer_model": f"{reviewer.provider}/{reviewer.model_name}",
                "model_settings": json.dumps({"primary": primary_client.settings, "second": second.settings, "reviewer": reviewer.settings}, sort_keys=True),
                "prompt_text": label_prompt + "\n\n--- REVIEW PROMPT ---\n\n" + reviewer_prompt,
                "prompt_version": f"label:{prompt_version(PROMPTS / 'label_conditions.md')};review:{prompt_version(PROMPTS / 'review_labels.md')}",
                "run_date": date.today().isoformat(),
            }
        )

    validation = pd.DataFrame(rows)
    universe = sorted(set().union(*primary_sets.values(), *secondary_sets.values()))
    left, right = [], []
    for item_id in primary_sets:
        for category in universe:
            left.append(int(category in primary_sets[item_id]))
            right.append(int(category in secondary_sets[item_id]))
    kappa = float(cohen_kappa_score(left, right)) if universe and len(set(left + right)) > 1 else float("nan")
    exact_agreement = float(validation["category_agreement"].mean()) if len(validation) else float("nan")
    repeat_rate = (
        sum(repeat_results.values()) / len(repeat_results) if repeat_results else float("nan")
    )
    unsupported = unsupported_labels / len(reviewed_labels) if reviewed_labels else float("nan")
    validation["cohens_kappa_overall"] = kappa
    validation["exact_item_agreement_overall"] = exact_agreement
    validation["repeat_agreement_overall"] = repeat_rate
    validation["unsupported_share_overall"] = unsupported
    validation["secondary_attempted_overall"] = len(agreement_sample)
    validation["secondary_failed_overall"] = len(secondary_failed)
    validation["repeat_attempted_overall"] = len(repeat_sample)
    validation["repeat_failed_overall"] = len(repeat_failed)
    validation["repeat_scored_overall"] = len(repeat_results)
    validation["review_attempted_overall"] = len(agreement_sample)
    validation["review_failed_overall"] = len(review_failed)
    validation["reviewed_labels_overall"] = len(reviewed_labels)
    validation["unsupported_labels_overall"] = unsupported_labels
    write_parquet_atomic(validation, DATA / "validation.parquet")
    _disagreement_html(validation, items)
    append_log(
        "Stage 4 validation results",
        f"Second-model exact tested-category agreement: {exact_agreement:.1%}  \nCohen's kappa: {kappa:.3f}  \n"
        f"Same-model repeat agreement: {repeat_rate:.1%}  \nReviewer unsupported share: {unsupported:.1%}  \n"
        f"Scored second-model items: {len(validation):,}/{len(agreement_sample):,}; "
        f"repeat failures: {len(repeat_failed):,}; reviewer items scored: "
        f"{len(successful_reviews):,}/{len(agreement_sample):,}; reviewer labels scored: "
        f"{len(reviewed_labels):,}; reviewer failures: {len(review_failed):,}",
    )
    print(f"Stage 4 complete: kappa {kappa:.3f}; exact category agreement {exact_agreement:.1%}; repeat agreement {repeat_rate:.1%}; unsupported {unsupported:.1%}.")
    return validation
