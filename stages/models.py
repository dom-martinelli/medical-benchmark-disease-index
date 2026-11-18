from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests

from .common import append_log, now_iso, update_metrics

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
# num_ctx 8192: the longest HealthBench prompts exceed Ollama's 4096 default, which
# silently truncates the start of the prompt (the instructions). num_predict 1024
# leaves room for items that name many conditions.
LOCAL_SETTINGS = {"temperature": 0, "seed": 20260911, "num_predict": 1024, "num_ctx": 8192}
GOOGLE_QUOTA_URL = "https://ai.google.dev/gemini-api/docs/rate-limits"


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response was valid JSON but not an object")
    return value


@dataclass
class ModelClient:
    provider: str
    model_name: str
    settings: dict[str, Any]
    free: bool = True
    api_key: str | None = None

    def generate_json(self, prompt: str) -> dict[str, Any]:
        started = time.monotonic()
        if self.provider == "ollama":
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                    "options": self.settings,
                },
                timeout=900,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["message"]["content"]
            input_tokens = int(payload.get("prompt_eval_count", 0))
            output_tokens = int(payload.get("eval_count", 0))
        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            response = requests.post(
                url,
                params={"key": self.api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.settings.get("temperature", 0),
                        "maxOutputTokens": self.settings.get("num_predict", 384),
                        "responseMimeType": "application/json",
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            usage = payload.get("usageMetadata", {})
            input_tokens = int(usage.get("promptTokenCount", 0))
            output_tokens = int(usage.get("candidatesTokenCount", 0))
        else:
            raise ValueError(f"Unknown model provider: {self.provider}")

        cost = 0.0
        if not self.free:
            input_price = float(os.environ["GEMINI_INPUT_USD_PER_MILLION"])
            output_price = float(os.environ["GEMINI_OUTPUT_USD_PER_MILLION"])
            cost = input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000
        update_metrics(
            api_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_wall_seconds=time.monotonic() - started,
        )
        return parse_json_object(text)


def _ollama_models() -> list[str]:
    response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    response.raise_for_status()
    return [str(x.get("name", "")) for x in response.json().get("models", [])]


def _start_ollama() -> bool:
    try:
        _ollama_models()
        return True
    except Exception:
        pass
    if not shutil.which("ollama"):
        return False
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            _ollama_models()
            return True
        except Exception:
            continue
    return False


def _ensure_ollama_model(model: str, allow_pull: bool) -> bool:
    installed = _ollama_models()
    if any(x == model or x.startswith(model + ":") or model.startswith(x + ":") for x in installed):
        return True
    if not allow_pull:
        return False
    print(f"Local model {model} is not installed. Downloading it with Ollama (free; disk and bandwidth required).")
    result = subprocess.run(["ollama", "pull", model], check=False)
    return result.returncode == 0


def _quota_snapshot() -> str:
    try:
        response = requests.get(GOOGLE_QUOTA_URL, timeout=30)
        response.raise_for_status()
        body = response.text
        text = re.sub(r"<[^>]+>", " ", body)
        text = html.unescape(re.sub(r"\s+", " ", text))
        relevant = " ".join(re.findall(r"[^.]{0,100}(?:RPD|requests per day|free tier)[^.]{0,180}", text, flags=re.I)[:8])
        return (
            f"Google rate-limit page read at runtime: {GOOGLE_QUOTA_URL}  \n"
            f"Retrieved: {now_iso()}  \nSHA-256: `{hashlib.sha256(body.encode()).hexdigest()}`  \n"
            f"Extracted quota text: {relevant[:1800] or 'Page retrieved, but no stable machine-readable daily cap was exposed.'}"
        )
    except Exception as exc:
        return f"Google rate-limit page could not be read at runtime: {type(exc).__name__}: {exc}"


def choose_primary_model(allow_paid: bool = False, allow_pull: bool = True) -> tuple[ModelClient, str]:
    configured = os.getenv("OLLAMA_MODEL")
    preferred_models = [configured] if configured else ["qwen2.5:7b-instruct", "gemma3:12b"]
    if _start_ollama():
        for preferred in preferred_models:
            try:
                if _ensure_ollama_model(preferred, allow_pull):
                    client = ModelClient(
                        "ollama",
                        preferred,
                        dict(LOCAL_SETTINGS),
                    )
                    return client, "Selected the first available option: a local Ollama model, with no API charge."
            except Exception as exc:
                append_log("Ollama detection warning", f"Model `{preferred}`: {type(exc).__name__}: {exc}")

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    paid_flag = os.getenv("GEMINI_PAID") == "1"
    if key and not paid_flag:
        append_log("Google AI Studio quota snapshot", _quota_snapshot())
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        return ModelClient("gemini", model, {"temperature": 0, "num_predict": 1024}, free=True, api_key=key), (
            "No working local model was available; selected a Google AI Studio key configured as free tier. "
            "The current quota page was read and recorded at run time."
        )
    if key and paid_flag:
        if not allow_paid:
            raise RuntimeError(
                "Only a paid Gemini configuration is available. Stop here and ask Dominic before spending money; "
                "then rerun with --allow-paid if he approves."
            )
        missing = [x for x in ("GEMINI_INPUT_USD_PER_MILLION", "GEMINI_OUTPUT_USD_PER_MILLION") if x not in os.environ]
        if missing:
            raise RuntimeError(f"Paid use was approved, but current pricing is not configured: missing {', '.join(missing)}")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ModelClient("gemini", model, {"temperature": 0, "num_predict": 1024}, free=False, api_key=key), "Selected paid Gemini only after the explicit paid-use flags were supplied."

    raise RuntimeError(
        "No usable model was found. Install/start Ollama, allow the pipeline to pull qwen2.5:7b-instruct, "
        "or set a free-tier GOOGLE_API_KEY/GEMINI_API_KEY from Google AI Studio."
    )


def choose_distinct_local_model(exclude: set[str], candidates: list[str], allow_pull: bool = True) -> ModelClient:
    if not _start_ollama():
        raise RuntimeError("Stage 4 needs distinct reviewer models, but Ollama is unavailable.")
    installed = _ollama_models()
    for candidate in candidates:
        matching = next((x for x in installed if x == candidate or x.startswith(candidate + ":")), None)
        selected = matching or candidate
        if selected in exclude:
            continue
        if matching or _ensure_ollama_model(candidate, allow_pull):
            return ModelClient(
                "ollama",
                candidate,
                dict(LOCAL_SETTINGS),
            )
    raise RuntimeError("Could not obtain a distinct local model for Stage 4 validation.")
