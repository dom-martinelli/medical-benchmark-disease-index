from __future__ import annotations

import os
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, get_token

from .common import DATA, ROOT, append_log


def _build_static_data(destination: Path) -> None:
    # The GitHub Pages build is the canonical static bundle. It contains the
    # harmonized item fields, benchmark catalog, and evaluation-availability
    # states produced by scripts/rebuild_public_explorer.py.
    canonical = ROOT / "docs" / "demo" / "data" / "explorer.json"
    if canonical.exists():
        shutil.copy2(canonical, destination)
        return
    coded = pd.read_parquet(DATA / "explorer_coded.parquet").fillna("")
    items = pd.read_parquet(DATA / "explorer_items.parquet").fillna("")
    rows_by_item: dict[str, list[list[str]]] = {}
    source_by_item: dict[str, str] = {}
    for row in coded.itertuples(index=False):
        item_id = str(row.item_id)
        source_by_item[item_id] = str(row.source)
        rows_by_item.setdefault(item_id, []).append(
            [
                str(row.condition_name),
                str(row.role),
                str(row.no_condition_category),
                str(row.code),
                str(row.ccsr_description),
                str(row.icd_chapter),
            ]
        )
    payload_items = []
    for row in items.itertuples(index=False):
        allowed = bool(row.question_redistribution_allowed)
        license_name = str(row.license or "unknown")
        try:
            answer_options = json.loads(str(row.answer_options or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            answer_options = {}
        try:
            wrong_option_map = json.loads(str(row.wrong_option_map or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            wrong_option_map = {}
        payload_items.append(
            {
                "id": str(row.item_id),
                "s": source_by_item.get(str(row.item_id), ""),
                "q": str(row.question),
                "a": (
                    f"{row.expected_answer_key} · {row.expected_answer}"
                    if str(row.expected_answer_key)
                    and str(row.expected_answer_key).casefold() != str(row.expected_answer).casefold()
                    else str(row.expected_answer)
                ),
                "k": str(row.expected_answer_key),
                "o": answer_options,
                "w": wrong_option_map,
                "n": (
                    f"Question shown; source card license: {license_name}"
                    if allowed
                    else f"Question withheld; source card license is {license_name}"
                ),
                "r": rows_by_item.get(str(row.item_id), []),
            }
        )
    clean = lambda values: sorted({str(x) for x in values if str(x) and str(x) != "unmatched"})
    payload = {
        "items": payload_items,
        "options": {
            "sources": clean(coded["source"]),
            "categories": clean(coded["ccsr_description"]),
            "chapters": clean(coded["icd_chapter"]),
            "roles": ["tested", "context", "ruled_out", "wrong_option"],
        },
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def run() -> str | None:
    # Prefer an explicit environment variable, then fall back to the token
    # securely stored by `hf auth login`. This avoids requiring users to paste
    # credentials into chat or keep them in project files.
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        append_log(
            "Hugging Face Space deployment deferred",
            "No `HF_TOKEN` environment variable or cached `hf auth login` token was found. "
            "No external repository was created or changed.",
        )
        print("Hugging Face deployment deferred: no local login token was found. The redacted Space bundle is ready locally.")
        return None
    required = [DATA / "explorer_coded.parquet", DATA / "explorer_items.parquet"]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Run Stage 5 report preparation before deployment; missing {', '.join(missing)}")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = os.getenv("HF_SPACE_ID", f"{user}/medical-benchmark-disease-index")
    space_sdk = os.getenv("HF_SPACE_SDK", "gradio").strip().lower()
    if space_sdk not in {"gradio", "static"}:
        raise ValueError("HF_SPACE_SDK must be either 'gradio' or 'static'.")
    api.create_repo(repo_id, repo_type="space", space_sdk=space_sdk, exist_ok=True, private=False)
    with tempfile.TemporaryDirectory(prefix="benchmark-space-") as temp_name:
        staging = Path(temp_name)
        (staging / "data").mkdir()
        if space_sdk == "static":
            shutil.copy2(ROOT / "docs" / "demo" / "index.html", staging / "index.html")
            _build_static_data(staging / "data" / "explorer.json")
            (staging / "README.md").write_text(
                "---\ntitle: Medical Benchmark Disease Index\nemoji: 🩺\ncolorFrom: green\ncolorTo: blue\n"
                "sdk: static\napp_file: index.html\npinned: false\n---\n\n"
                "A filterable, redacted index of disease coverage in public medical AI benchmarks.\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(ROOT / "app.py", staging / "app.py")
            shutil.copy2(ROOT / "space-requirements.txt", staging / "requirements.txt")
            shutil.copy2(ROOT / "SPACE_README.md", staging / "README.md")
            shutil.copy2(DATA / "explorer_coded.parquet", staging / "data" / "explorer_coded.parquet")
            shutil.copy2(DATA / "explorer_items.parquet", staging / "data" / "explorer_items.parquet")
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=staging, commit_message="Publish disease benchmark explorer")
    url = f"https://huggingface.co/spaces/{repo_id}"
    append_log("Hugging Face Space deployed", f"Space: {url}  \nOnly redacted explorer data was uploaded.")
    print(f"Deployed the redacted {space_sdk} explorer to {url}")
    return url
