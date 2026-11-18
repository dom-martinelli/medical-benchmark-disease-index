from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterator

import pandas as pd
import requests
from urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .common import CHECKPOINTS, DATA, append_jsonl, append_log, ensure_dirs, read_jsonl, write_parquet_atomic

VIEWER = "https://datasets-server.huggingface.co"
HF_API = "https://huggingface.co/api/datasets"
HEALTHBENCH_URL = (
    "https://huggingface.co/datasets/openai/healthbench/resolve/main/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    splits: tuple[str, ...]
    parser: Callable[[dict[str, Any], str], dict[str, Any]]
    raw_url: str | None = None


def _options(value: Any) -> str:
    if value is None or (isinstance(value, str) and value == ""):
        return "[]"
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "key" in value[0]:
            value = {str(x["key"]): x.get("value", "") for x in value}
        else:
            value = {chr(65 + i): x for i, x in enumerate(value)}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _base(question: str, options: Any, answer: Any, explanation: Any) -> dict[str, Any]:
    answer_text = "" if answer is None else str(answer)
    return {
        "question": str(question or "").strip(),
        "options": _options(options),
        "answer_key": answer_text,
        "explanation": str(explanation or "").strip(),
        "has_answer_key": bool(answer_text.strip()),
    }


def parse_medqa(row: dict[str, Any], split: str) -> dict[str, Any]:
    return _base(row.get("question"), row.get("options"), row.get("answer_idx"), "")


def parse_medmcqa(row: dict[str, Any], split: str) -> dict[str, Any]:
    options = {"A": row.get("opa", ""), "B": row.get("opb", ""), "C": row.get("opc", ""), "D": row.get("opd", "")}
    raw = row.get("cop")
    answer = chr(65 + int(raw)) if raw is not None and str(raw) != "" else ""
    parsed = _base(row.get("question"), options, answer, row.get("exp"))
    parsed["source_record_id"] = str(row.get("id", ""))
    return parsed


def parse_pubmedqa(row: dict[str, Any], split: str) -> dict[str, Any]:
    parsed = _base(row.get("question"), {"A": "yes", "B": "no", "C": "maybe"}, row.get("final_decision"), row.get("long_answer"))
    parsed["source_record_id"] = str(row.get("pubid", ""))
    return parsed


def parse_mmlu(row: dict[str, Any], split: str) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, int):
        answer = chr(65 + answer)
    return _base(row.get("question"), row.get("choices"), answer, "")


def parse_medbullets(row: dict[str, Any], split: str) -> dict[str, Any]:
    parsed = _base(row.get("question"), row.get("options"), row.get("answer"), row.get("explanation"))
    parsed["source_record_id"] = str(row.get("idx", ""))
    return parsed


def parse_healthbench(row: dict[str, Any], split: str) -> dict[str, Any]:
    prompt = row.get("prompt") or []
    # Keep every turn with its speaker. 2,085 of 5,000 prompts are multi-turn, and
    # the final user message often only makes sense after the assistant's reply.
    turns = [x for x in prompt if str(x.get("content", "")).strip()]
    if len(turns) == 1 and turns[0].get("role") == "user":
        question = str(turns[0].get("content", ""))
    else:
        question = "\n\n".join(f"[{str(x.get('role', 'user')).upper()}] {x.get('content', '')}" for x in turns)
    rubrics = row.get("rubrics") or []
    explanation = "\n".join(str(x.get("criterion", "")) for x in rubrics)
    parsed = _base(question, [], "", explanation)
    parsed["source_record_id"] = str(row.get("prompt_id", ""))
    return parsed


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("MedQA", "bigbio/med_qa", "med_qa_en_4options_source", ("test", "validation"), parse_medqa),
    SourceSpec("MedMCQA", "openlifescienceai/medmcqa", "default", ("validation",), parse_medmcqa),
    # The labeled PubMedQA subset exposes only a train split. This documented exception is retained.
    SourceSpec("PubMedQA", "qiaojin/PubMedQA", "pqa_labeled", ("train",), parse_pubmedqa),
    *tuple(
        SourceSpec(f"MMLU {cfg.replace('_', ' ')}", "cais/mmlu", cfg, ("test", "validation"), parse_mmlu)
        for cfg in ("anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine")
    ),
    # op4_test and op5_test are two derived versions of the same 308 parent questions.
    SourceSpec("MedBullets", "mkieffer/Medbullets", "default", ("op5_test",), parse_medbullets),
    SourceSpec("HealthBench", "openai/healthbench", "default", ("test",), parse_healthbench, HEALTHBENCH_URL),
)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "benchmark-disease-index/1.0"})
    retry = Retry(total=6, connect=4, read=4, status=6, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504), respect_retry_after_header=True)
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _parquet_rows(session: requests.Session, spec: SourceSpec, split: str, skip: int = 0) -> Iterator[dict[str, Any]]:
    response = session.get(f"{VIEWER}/parquet", params={"dataset": spec.dataset}, timeout=120)
    response.raise_for_status()
    files = [
        x for x in response.json().get("parquet_files", [])
        if x.get("config") == spec.config and x.get("split") == split
    ]
    if not files:
        raise FileNotFoundError(f"No Parquet export found for {spec.dataset}/{spec.config}/{split}")
    seen = 0
    cache_dir = DATA / "reference" / "hf_parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for entry in files:
        url = str(entry["url"])
        cache = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:20]}.parquet"
        if not cache.exists():
            with session.get(url, stream=True, timeout=(30, 300)) as download:
                download.raise_for_status()
                temp = cache.with_suffix(".parquet.tmp")
                with temp.open("wb") as handle:
                    for chunk in download.iter_content(1024 * 1024):
                        handle.write(chunk)
                temp.replace(cache)
        frame = pd.read_parquet(cache)
        for row in frame.to_dict("records"):
            if seen < skip:
                seen += 1
                continue
            yield row


def _metadata(session: requests.Session, dataset: str) -> dict[str, Any]:
    response = session.get(f"{HF_API}/{dataset}", timeout=60)
    response.raise_for_status()
    obj = response.json()
    card = obj.get("cardData") or {}
    license_value = card.get("license")
    if isinstance(license_value, list):
        license_value = ", ".join(str(x) for x in license_value)
    return {
        "dataset": dataset,
        "revision": obj.get("sha", "unknown"),
        "license": str(license_value or "unknown"),
        "gated": bool(obj.get("gated")),
    }


def _viewer_rows(session: requests.Session, spec: SourceSpec, split: str, start_offset: int = 0) -> Iterator[dict[str, Any]]:
    offset = start_offset
    total: int | None = None
    while total is None or offset < total:
        response = session.get(
            f"{VIEWER}/rows",
            params={"dataset": spec.dataset, "config": spec.config, "split": split, "offset": offset, "length": 100},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])
        total = int(payload.get("num_rows_total", offset + len(rows)))
        if not rows:
            break
        for wrapped in rows:
            yield wrapped.get("row", wrapped)
        offset += len(rows)


def _raw_jsonl_rows(session: requests.Session, url: str, skip: int = 0) -> Iterator[dict[str, Any]]:
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        seen = 0
        for line in response.iter_lines(decode_unicode=True):
            if line and line.strip():
                if seen < skip:
                    seen += 1
                    continue
                yield json.loads(line)


def _stable_id(source: str, split: str, question: str, source_record_id: str = "") -> str:
    basis = source_record_id or re.sub(r"\s+", " ", question).strip().casefold()
    digest = hashlib.sha256(f"{source}|{split}|{basis}".encode()).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")
    return f"{slug}-{digest}"


def _redistribution_allowed(license_name: str, source: str = "") -> bool:
    # HealthBench's maintainers explicitly request that examples not be exposed
    # publicly, to reduce leakage into model training corpora. Honor that
    # source-specific request independently of the repository's MIT license.
    if source.strip().casefold() == "healthbench":
        return False
    # Hugging Face mirror metadata can lose the parent's license. MedQA's
    # bigbio mirror says "unknown", but the original release (jind11/MedQA) is
    # MIT, and the mirror's text is identical to it (lineage audit 2026-09-14).
    # Medbullets has no license on the mirror or on HanjieChen/ChallengeClinicalQA,
    # so it stays withheld.
    if source.strip().casefold() == "medqa":
        return True
    values = {x.strip().casefold() for x in license_name.split(",")}
    return bool(values & {"mit", "apache-2.0", "apache 2.0", "cc-by-4.0", "cc0-1.0"})


def _flag_near_duplicates(df: pd.DataFrame, threshold: float = 0.94) -> pd.DataFrame:
    if len(df) < 2:
        df["near_duplicate_item_ids"] = "[]"
        df["max_cross_source_similarity"] = 0.0
        return df
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(df["question"].fillna(""))
    count = min(25, len(df))
    distances, indices = NearestNeighbors(n_neighbors=count, metric="cosine", n_jobs=-1).fit(matrix).kneighbors(matrix)
    ids = df["item_id"].tolist()
    sources = df["source"].tolist()
    pairs: list[list[str]] = [[] for _ in ids]
    maxima = [0.0 for _ in ids]
    for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, j in zip(row_distances[1:], row_indices[1:]):
            if sources[i] == sources[j]:
                continue
            similarity = 1.0 - float(distance)
            maxima[i] = max(maxima[i], similarity)
            if similarity >= threshold:
                pairs[i].append(ids[j])
    df["near_duplicate_item_ids"] = [json.dumps(sorted(set(x))) for x in pairs]
    df["max_cross_source_similarity"] = maxima
    return df


def run(limit: int | None = None) -> pd.DataFrame:
    ensure_dirs()
    session = _session()
    retrieved_on = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    metadata_cache: dict[str, dict[str, Any]] = {}
    remaining = limit

    for spec in SOURCES:
        try:
            meta = metadata_cache.setdefault(spec.dataset, _metadata(session, spec.dataset))
        except Exception as exc:
            reports.append({"source": spec.name, "split": "all", "count": 0, "status": f"skipped: metadata failed: {exc}"})
            continue
        for split in spec.splits:
            if remaining is not None and remaining <= 0:
                break
            checkpoint_key = hashlib.sha256(f"{spec.dataset}|{meta['revision']}|{spec.config}|{split}".encode()).hexdigest()[:12]
            checkpoint = CHECKPOINTS / f"collect-{checkpoint_key}.jsonl"
            done_marker = CHECKPOINTS / f"collect-{checkpoint_key}.done"
            saved = read_jsonl(checkpoint)
            if remaining is not None and len(saved) >= remaining:
                rows.extend(saved[:remaining])
                reports.append({"source": spec.name, "split": split, "count": remaining, "status": "limited smoke-test collection from checkpoint"})
                remaining = 0
                break
            rows.extend(saved)
            if remaining is not None:
                remaining -= len(saved)
            count = len(saved)
            if done_marker.exists():
                reports.append({"source": spec.name, "split": split, "count": count, "status": "collected (checkpoint reused)"})
                continue
            status = "collected"
            try:
                raw_url = spec.raw_url.replace("/resolve/main/", f"/resolve/{meta['revision']}/") if spec.raw_url else None
                iterator = _raw_jsonl_rows(session, raw_url, skip=count) if raw_url else _parquet_rows(session, spec, split, skip=count)
                for raw in iterator:
                    parsed = spec.parser(raw, split)
                    if not parsed["question"]:
                        continue
                    source_record_id = parsed.pop("source_record_id", "")
                    item = {
                        "item_id": _stable_id(spec.name, split, parsed["question"], source_record_id),
                        "source": spec.name,
                        "split": split,
                        **parsed,
                        "license": meta["license"],
                        "question_redistribution_allowed": _redistribution_allowed(meta["license"], spec.name),
                        "retrieved_on": retrieved_on,
                        "dataset_id": spec.dataset,
                        "dataset_revision": meta["revision"],
                        "prompt_text": "",
                        "model_name": "none",
                        "model_settings": "{}",
                    }
                    rows.append(item)
                    append_jsonl(checkpoint, [item])
                    count += 1
                    if len(rows) % 100 == 0:
                        print(f"Collected {len(rows):,} items; currently reading {spec.name} ({split}).")
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            status = "limited smoke-test collection"
                            break
                if status == "collected":
                    done_marker.write_text(f"completed {retrieved_on}\n", encoding="utf-8")
            except Exception as exc:
                status = f"skipped/partial after {count}: {type(exc).__name__}: {exc}"
            reports.append({"source": spec.name, "split": split, "count": count, "status": status})

    if not rows:
        raise RuntimeError("No benchmark items were collected. See network/source errors above.")
    df = pd.DataFrame(rows).reset_index(drop=True)
    # A few source files repeat identical question text without exposing a record ID.
    # Retain every source row and make colliding IDs deterministic rather than dropping data.
    collision_number = df.groupby("item_id", sort=False).cumcount()
    collided = collision_number > 0
    df.loc[collided, "item_id"] = df.loc[collided, "item_id"] + "-repeat-" + collision_number[collided].astype(str)
    # The BigBio `answer` field is the correct option text, not an explanation.
    # Older collection checkpoints may predate that distinction, so normalize it here.
    df.loc[df["source"] == "MedQA", "explanation"] = ""
    # Reapply the current publication rule even when source rows came from an
    # older checkpoint. This lets source-specific requests become effective
    # without redownloading otherwise unchanged benchmark data.
    df["question_redistribution_allowed"] = [
        _redistribution_allowed(str(license_name), str(source))
        for license_name, source in zip(df["license"], df["source"])
    ]
    df = _flag_near_duplicates(df)
    ordered = [
        "item_id", "source", "split", "question", "options", "answer_key", "explanation",
        "has_answer_key", "license", "retrieved_on", "near_duplicate_item_ids",
        "max_cross_source_similarity", "question_redistribution_allowed", "dataset_id",
        "dataset_revision", "prompt_text", "model_name", "model_settings",
    ]
    df = df[ordered]
    write_parquet_atomic(df, DATA / "items.parquet")
    (DATA / "source_metadata.json").write_text(json.dumps(list(metadata_cache.values()), indent=2), encoding="utf-8")

    counts = pd.DataFrame(reports)
    report_lines = [
        "# Collection report",
        "",
        f"Retrieved {len(df):,} unique benchmark items on {retrieved_on}. "
        f"{int(df['has_answer_key'].sum()):,} have a public answer key and {int((~df['has_answer_key']).sum()):,} do not.",
        "",
        counts.to_markdown(index=False),
        "",
        "PubMedQA's labeled subset exposes only a train split, so it is included as a documented exception to the test/validation rule. "
        "MedBullets op4 and op5 are derived versions of the same parent questions; only op5 is retained so explanations attach to one parent row. "
        "Sources with unknown or absent license metadata are retained, but their question text is suppressed in the explorer. "
        "HealthBench text is also suppressed to honor the source maintainers' request not to reproduce examples publicly.",
        "",
        f"Near-duplicate flagging found {int((df['near_duplicate_item_ids'] != '[]').sum()):,} rows with at least one cross-source text match at cosine similarity 0.94 or higher. Both rows are retained.",
    ]
    if limit is not None:
        report_lines.extend(["", f"**Validation warning:** this was a smoke-test run capped at {limit:,} total items; these are not paper-ready counts."])
    (DATA / "collection_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    append_log("Stage 1 collection", f"Items written: {len(df):,}  \nLimited run: `{limit is not None}`\n\n{counts.to_markdown(index=False)}")
    print(f"Stage 1 complete: wrote {len(df):,} rows to data/items.parquet.")
    return df
