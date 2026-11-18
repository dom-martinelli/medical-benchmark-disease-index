"""External agreement check: our ICD codes vs MedMCQA's own subject labels.

No model and no human. MedMCQA ships a subject for every item (Dental,
Ophthalmology, ...). For subjects tied to one body system, a correct tested
condition should usually carry a code in that system's ICD range. The expected
ranges are frozen in prompts/external_check_subject_map.json before any result
is looked at. Agreement is reported against a permutation baseline, because the
subject mix alone (Dental is a third of the split) sets a floor that is not zero.

Reads labels from the Stage 2 checkpoint, so it can run while labeling is still
going on other sources. Writes only its own outputs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .code import CodeLookup, load_reference
from .collect import _stable_id
from .common import DATA, FIGURES, PROMPTS, REPORT, append_log, ensure_dirs, prompt_version, read_jsonl
from .label import CHECKPOINT, FAILED, PROMPT_PATH, keep_latest_outcome
from .models import LOCAL_SETTINGS

MAP_PATH = PROMPTS / "external_check_subject_map.json"
MEDMCQA_PARQUET_GLOB = "data/reference/hf_parquet/*.parquet"
PERMUTATIONS = 1000
SEED = 20260914


def _parse_range(spec: str) -> tuple[str, str]:
    low, _, high = spec.partition("-")
    return low, high or low


def _in_ranges(code: str, ranges: list[str]) -> bool:
    raw = code.replace(".", "").upper()
    for spec in ranges:
        low, high = _parse_range(spec)
        width = max(len(low), len(high))
        head = raw[:width]
        if low <= head <= high:
            return True
    return False


def _wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _medmcqa_subjects() -> pd.DataFrame:
    from glob import glob

    for path in glob(str(Path(__file__).resolve().parents[1] / MEDMCQA_PARQUET_GLOB)):
        frame = pd.read_parquet(path)
        if {"subject_name", "opa", "cop"}.issubset(frame.columns):
            frame = frame.copy()
            frame["item_id"] = [
                _stable_id("MedMCQA", "validation", str(q), str(i)) for q, i in zip(frame["question"], frame["id"])
            ]
            return frame[["item_id", "subject_name"]]
    raise FileNotFoundError("Cached MedMCQA validation parquet with subject_name was not found; run Stage 1 first.")


def _current_labels(source: str) -> pd.DataFrame:
    settings = json.dumps(LOCAL_SETTINGS, sort_keys=True)
    version = prompt_version(PROMPT_PATH)
    rows = [
        r for r in read_jsonl(CHECKPOINT)
        if r.get("source") == source and r.get("prompt_version") == version and r.get("model_settings") == settings
    ]
    return pd.DataFrame(keep_latest_outcome(rows))


def run() -> dict[str, Any]:
    ensure_dirs()
    spec = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    subjects = _medmcqa_subjects()
    labels = _current_labels("MedMCQA")
    if labels.empty:
        raise RuntimeError("No MedMCQA labels with the current prompt and settings are in the Stage 2 checkpoint yet.")

    reference, _ = load_reference()
    lookup = CodeLookup(reference)
    names = sorted({str(x).strip() for x in labels["condition_name"] if str(x).strip()})
    coded = {name: lookup.match(name) for name in names}
    labels["code"] = labels["condition_name"].map(lambda x: coded.get(str(x).strip(), {}).get("code", "unmatched"))

    items = subjects.merge(
        labels.groupby("item_id").agg(
            tested_codes=("code", lambda s: [c for c, r in zip(s, labels.loc[s.index, "role"]) if r == "tested" and c != "unmatched"]),
            any_codes=("code", lambda s: [c for c in s if c != "unmatched"]),
            no_condition=("no_condition_category", lambda s: bool((s != "").all() and (s != FAILED).all())),
            failed=("no_condition_category", lambda s: bool((s == FAILED).any())),
        ).reset_index(),
        on="item_id", how="left",
    )
    joined = int(items["tested_codes"].notna().sum())

    rows, per_subject = [], []
    mapped = spec["subjects"]
    for subject, ranges in mapped.items():
        strict = ranges["strict"]
        lenient = strict + ranges["lenient_adds"]
        frame = items[items["subject_name"] == subject]
        have = frame[frame["tested_codes"].map(lambda x: isinstance(x, list) and len(x) > 0)]
        strict_hits = int(have["tested_codes"].map(lambda codes: any(_in_ranges(c, strict) for c in codes)).sum())
        lenient_hits = int(have["tested_codes"].map(lambda codes: any(_in_ranges(c, lenient) for c in codes)).sum())
        low, high = _wilson(strict_hits, len(have))
        per_subject.append({
            "subject": subject, "items": len(frame), "with_coded_tested_condition": len(have),
            "strict_agree": strict_hits, "strict_rate": strict_hits / len(have) if len(have) else float("nan"),
            "strict_ci_low": low, "strict_ci_high": high,
            "lenient_agree": lenient_hits, "lenient_rate": lenient_hits / len(have) if len(have) else float("nan"),
        })
        for r in have.itertuples():
            rows.append({"item_id": r.item_id, "subject": subject, "codes": r.tested_codes,
                         "strict": any(_in_ranges(c, strict) for c in r.tested_codes),
                         "lenient": any(_in_ranges(c, lenient) for c in r.tested_codes)})

    detail = pd.DataFrame(rows)
    table = pd.DataFrame(per_subject)
    n = len(detail)
    observed_strict = float(detail["strict"].mean()) if n else float("nan")
    observed_lenient = float(detail["lenient"].mean()) if n else float("nan")

    # Permutation baseline: shuffle subject labels across the scored items and
    # recompute agreement. Keeps each subject's item count and each item's codes.
    rng = np.random.default_rng(SEED)
    subject_array = detail["subject"].to_numpy()
    codes_list = detail["codes"].tolist()
    strict_by_subject = {s: mapped[s]["strict"] for s in mapped}
    lenient_by_subject = {s: mapped[s]["strict"] + mapped[s]["lenient_adds"] for s in mapped}
    hit_strict = {s: np.array([any(_in_ranges(c, strict_by_subject[s]) for c in codes) for codes in codes_list]) for s in mapped}
    hit_lenient = {s: np.array([any(_in_ranges(c, lenient_by_subject[s]) for c in codes) for codes in codes_list]) for s in mapped}
    null_strict, null_lenient = [], []
    index = np.arange(n)
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(subject_array)
        null_strict.append(float(np.mean([hit_strict[s][i] for i, s in zip(index, shuffled)])))
        null_lenient.append(float(np.mean([hit_lenient[s][i] for i, s in zip(index, shuffled)])))
    chance_strict = float(np.mean(null_strict))
    chance_lenient = float(np.mean(null_lenient))
    p_strict = (1 + sum(x >= observed_strict for x in null_strict)) / (PERMUTATIONS + 1)

    # Cohen-style chance-corrected agreement: how far above chance, as a share of the room above chance.
    kappa_like = (observed_strict - chance_strict) / (1 - chance_strict) if chance_strict < 1 else float("nan")

    ped = spec["pediatrics_enrichment"]
    def share_with(subject_list: list[str]) -> tuple[int, int]:
        frame = items[items["subject_name"].isin(subject_list)]
        have = frame[frame["any_codes"].map(lambda x: isinstance(x, list) and len(x) > 0)]
        hits = int(have["any_codes"].map(lambda codes: any(_in_ranges(c, ped["codes"]) for c in codes)).sum())
        return hits, len(have)
    ped_hits, ped_n = share_with([ped["subject"]])
    cmp_hits, cmp_n = share_with(ped["comparison_subjects"])

    basic = spec["basic_science_check"]
    def no_condition_rate(subject_list: list[str]) -> tuple[int, int]:
        frame = items[items["subject_name"].isin(subject_list) & items["tested_codes"].notna() & ~items["failed"].fillna(False)]
        return int(frame["no_condition"].sum()), len(frame)
    basic_hits, basic_n = no_condition_rate(basic["basic_subjects"])
    clin_hits, clin_n = no_condition_rate(basic["clinical_subjects"])

    ensure_dirs()
    out_dir = DATA / "external_check"
    out_dir.mkdir(exist_ok=True)
    detail.assign(codes=detail["codes"].map(json.dumps)).to_parquet(out_dir / "medmcqa_subject_agreement.parquet", index=False)
    table.to_csv(out_dir / "medmcqa_subject_agreement_by_subject.csv", index=False)

    _figure(table, chance_strict)
    summary = {
        "map_version": spec["version"], "prompt_version": prompt_version(PROMPT_PATH),
        "medmcqa_items": len(subjects), "items_with_labels": joined, "scored_items": n,
        "observed_strict": observed_strict, "observed_lenient": observed_lenient,
        "chance_strict": chance_strict, "chance_lenient": chance_lenient,
        "chance_corrected_strict": kappa_like, "permutation_p_strict": p_strict,
        "pediatrics_pq_share": [ped_hits, ped_n], "medicine_surgery_pq_share": [cmp_hits, cmp_n],
        "basic_no_condition": [basic_hits, basic_n], "clinical_no_condition": [clin_hits, clin_n],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, table)
    append_log(
        "External check: MedMCQA subject agreement",
        f"Subject map version: `{spec['version']}` (frozen before results)  \nScored items: {n:,}  \n"
        f"Strict agreement {observed_strict:.1%} vs chance {chance_strict:.1%} (p={p_strict:.3f}); lenient {observed_lenient:.1%} vs {chance_lenient:.1%}",
    )
    print(f"External check: {n:,} MedMCQA items scored; strict agreement {observed_strict:.1%} vs chance {chance_strict:.1%}.")
    return summary


def _figure(table: pd.DataFrame, chance: float) -> None:
    import matplotlib.pyplot as plt

    frame = table.sort_values("with_coded_tested_condition", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(frame))
    ax.barh(y, frame["lenient_rate"] * 100, color="#c9d9f2", label="lenient range")
    ax.barh(y, frame["strict_rate"] * 100, color="#35558f", label="strict range")
    err_low = (frame["strict_rate"] - frame["strict_ci_low"]) * 100
    err_high = (frame["strict_ci_high"] - frame["strict_rate"]) * 100
    ax.errorbar(frame["strict_rate"] * 100, y, xerr=[err_low, err_high], fmt="none", ecolor="#1d2430", capsize=3, lw=1)
    ax.axvline(chance * 100, color="#d9705e", ls="--", lw=1.5, label=f"chance, strict ({chance:.0%})")
    ax.set_yticks(y, [f"{s}  (n={n})" for s, n in zip(frame["subject"], frame["with_coded_tested_condition"])])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Items whose tested condition's ICD code falls in the subject's expected range (%)")
    ax.set_title("Our ICD codes vs MedMCQA's own subject labels")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES / "external_check_medmcqa_subject.png", dpi=200)
    plt.close()


def _write_report(s: dict[str, Any], table: pd.DataFrame) -> None:
    ped_h, ped_n = s["pediatrics_pq_share"]
    cmp_h, cmp_n = s["medicine_surgery_pq_share"]
    bas_h, bas_n = s["basic_no_condition"]
    cli_h, cli_n = s["clinical_no_condition"]
    pretty = table.assign(
        strict=lambda d: [f"{r:.0%} ({lo:.0%}–{hi:.0%})" if n else "—" for r, lo, hi, n in zip(d.strict_rate, d.strict_ci_low, d.strict_ci_high, d.with_coded_tested_condition)],
        lenient=lambda d: [f"{r:.0%}" if n else "—" for r, n in zip(d.lenient_rate, d.with_coded_tested_condition)],
    )[["subject", "items", "with_coded_tested_condition", "strict", "lenient"]]
    text = f"""# External agreement check: ICD codes vs MedMCQA subject labels

MedMCQA assigns every item a subject. For nine subjects tied to one body system, the expected ICD-10-CM ranges were written down and frozen in `prompts/external_check_subject_map.json` (version {s['map_version']}) before any comparison was run. An item agrees when at least one of its coded `tested` conditions falls in its subject's range. No model and no human were involved in this check.

Of {s['medmcqa_items']:,} MedMCQA validation items, {s['items_with_labels']:,} had labels from prompt {s['prompt_version']}, and {s['scored_items']:,} items in the nine subjects had at least one coded tested condition and were scored.

## Result

Strict agreement was **{s['observed_strict']:.1%}**, against **{s['chance_strict']:.1%}** expected by chance (subject labels shuffled across scored items {PERMUTATIONS:,} times; permutation p = {s['permutation_p_strict']:.3f}). Chance-corrected, that is {s['chance_corrected_strict']:.2f} of the possible improvement over chance. With lenient ranges, agreement was {s['observed_lenient']:.1%} against {s['chance_lenient']:.1%}.

{pretty.to_markdown(index=False)}

Strict intervals are Wilson 95% intervals. Subjects with fewer than about 30 scored items are too small to read individually.

## Two supporting checks

**Pediatrics.** Among Pediatrics items with any coded condition, {ped_h}/{ped_n} ({ped_h / ped_n if ped_n else float('nan'):.1%}) carried a perinatal or congenital code (ICD P or Q), compared with {cmp_h}/{cmp_n} ({cmp_h / cmp_n if cmp_n else float('nan'):.1%}) for Medicine and Surgery.

**Basic science.** Anatomy, Physiology, and Biochemistry items were labeled as containing no condition {bas_h}/{bas_n} times ({bas_h / bas_n if bas_n else float('nan'):.1%}), compared with {cli_h}/{cli_n} ({cli_h / cli_n if cli_n else float('nan'):.1%}) for clinical subjects.

## What this does and does not show

Agreement here means the labeler put conditions in the right body system. It does not show that the specific condition or its role is correct, and a subject label is itself a coarse, author-assigned tag: a Dental item can legitimately test a systemic disease. The check covers only MedMCQA, and the subject mix limits which ICD chapters it can speak to. It is an external plausibility check, not an accuracy estimate.
"""
    (REPORT / "external_check.md").write_text(text, encoding="utf-8")
