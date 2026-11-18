from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process

from .common import (
    CHECKPOINTS, DATA, PROMPTS, append_jsonl, append_log, ensure_dirs, now_iso, prompt_version, read_jsonl, sha256_text,
    write_parquet_atomic,
)

ICD_URL = "https://www.cms.gov/files/zip/april-1-2026-code-descriptions-tabular-order.zip"
CCSR_URL = "https://hcup-us.ahrq.gov/toolssoftware/ccsr/DXCCSR-v2026-1.zip"
INDEX_URL = "https://www.cms.gov/files/zip/april-1-2026-code-tables-tabular-index.zip"
ICD_VERSION = "FY2026 April 1, 2026 update"
CCSR_VERSION = "v2026.1"
LOOKUP_VERSION = "5"
REFERENCE_DIR = DATA / "reference"
MAPPING_CHECKPOINT = CHECKPOINTS / "condition_code_map.jsonl"
RERANK_CHECKPOINT = CHECKPOINTS / "code_rerank.jsonl"
RERANK_PROMPT = PROMPTS / "code_rerank.md"
EXPANSIONS_PATH = PROMPTS / "condition_abbreviations.json"
# Only these lookup tiers are sent to the optional model rerank (CODE_RERANK=1);
# exact, index, and Neoplasm Table matches are the official wording already.
RERANK_METHODS = {"fuzzy", "embedding", "unmatched"}


def _download(url: str, path: Path) -> str:
    if not path.exists():
        print(f"Downloading pinned reference file {path.name}.")
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            temp = path.with_suffix(path.suffix + ".tmp")
            with temp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_header(value: str) -> str:
    return value.strip().strip("'").strip('"')


# Duplicate 7th-character variants carry no extra disease information; the
# "initial encounter" variant is kept as the representative code.
_EPISODE_DUPLICATE = re.compile(r"subsequent encounter|sequela|delayed healing|nonunion|malunion|routine healing", re.I)


def _load_ccsr(ccsr_path: Path) -> dict[str, tuple[str, str]]:
    with zipfile.ZipFile(ccsr_path) as archive:
        csv_name = next(x for x in archive.namelist() if x.lower().endswith(".csv"))
        raw = archive.read(csv_name).decode("utf-8-sig", "replace")
    output: dict[str, tuple[str, str]] = {}
    for original in csv.DictReader(io.StringIO(raw), quotechar='"'):
        row = {_clean_header(k): (v or "").strip().strip("'") for k, v in original.items() if k is not None}
        code = row.get("ICD-10-CM CODE", "").replace(".", "").strip()
        if not code:
            continue
        # XXX-prefixed values are AHRQ placeholders (e.g. "unacceptable principal
        # diagnosis"), not clinical categories; fall through to the next real one.
        pairs = [
            (row.get("Default CCSR CATEGORY OP", ""), row.get("Default CCSR CATEGORY DESCRIPTION OP", "")),
            (row.get("Default CCSR CATEGORY IP", ""), row.get("Default CCSR CATEGORY DESCRIPTION IP", "")),
        ] + [(row.get(f"CCSR CATEGORY {n}", ""), row.get(f"CCSR CATEGORY {n} DESCRIPTION", "")) for n in range(1, 7)]
        real = next(((c, d) for c, d in pairs if c and not c.upper().startswith("XXX")), None)
        if real:
            output[code] = real
    return output


def load_reference() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every ICD-10-CM code in the official order file, including non-billable
    category headers (e.g. G30 Alzheimer's disease), each with a CCSR roll-up.
    Headers take the most common CCSR category of their billable descendants."""
    ensure_dirs()
    icd_path = REFERENCE_DIR / "icd10cm-April-1-2026.zip"
    ccsr_path = REFERENCE_DIR / "DXCCSR-v2026-1.zip"
    icd_sha = _download(ICD_URL, icd_path)
    ccsr_sha = _download(CCSR_URL, ccsr_path)
    ccsr = _load_ccsr(ccsr_path)

    with zipfile.ZipFile(icd_path) as archive:
        order_name = next(x for x in archive.namelist() if re.search(r"icd10cm_order_\d{4}\.txt$", x))
        lines = archive.read(order_name).decode("utf-8", "replace").splitlines()
    rows = []
    for line in lines:
        if len(line) < 16:
            continue
        code, billable, description = line[6:13].strip(), line[14] == "1", line[77:].strip() or line[16:76].strip()
        if not code or _EPISODE_DUPLICATE.search(description):
            continue
        rows.append({"code_raw": code, "billable": billable, "description": description})
    reference = pd.DataFrame(rows).drop_duplicates("code_raw").reset_index(drop=True)

    votes: dict[str, dict[tuple[str, str], int]] = {}
    for code, value in ccsr.items():
        for length in range(3, len(code)):
            bucket = votes.setdefault(code[:length], {})
            bucket[value] = bucket.get(value, 0) + 1
    categories, descriptions = [], []
    for code in reference["code_raw"]:
        value = ccsr.get(code)
        if value is None and code in votes:
            value = max(votes[code].items(), key=lambda kv: (kv[1], kv[0]))[0]
        categories.append(value[0] if value else "unmatched")
        descriptions.append(value[1] if value else "Unmatched")
    reference["ccsr_category"] = categories
    reference["ccsr_description"] = descriptions

    manifest = {
        "icd_version": ICD_VERSION,
        "icd_url": ICD_URL,
        "icd_sha256": icd_sha,
        "icd_file_used": order_name,
        "ccsr_version": CCSR_VERSION,
        "ccsr_url": CCSR_URL,
        "ccsr_sha256": ccsr_sha,
        "lookup_version": LOOKUP_VERSION,
        "index_url": INDEX_URL,
        "index_sha256": _download(INDEX_URL, REFERENCE_DIR / "icd10cm-April-1-2026-tables-index.zip"),
        "reference_rows": len(reference),
    }
    (REFERENCE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return reference, manifest


def _index_title_parts(element) -> tuple[list[str], list[str]]:
    """Split an index title into its term variants and its nonessential modifiers."""
    title = element.find("title")
    if title is None:
        return [], []
    nemod = title.find("nemod")
    modifiers = re.findall(r"\(([^)]*)\)", nemod.text or "") if nemod is not None else []
    head = (title.text or "").strip()
    parts = [x.strip() for x in head.split(",") if x.strip()]
    # "Hypertension, hypertensive" lists word forms of one term, but
    # "Converter, tuberculosis" is an inverted phrase. Only parts sharing a stem
    # with the headword are variants; otherwise the phrase is re-read in order.
    if len(parts) > 1 and not all(x[:4].casefold() == parts[0][:4].casefold() for x in parts[1:]):
        parts = [" ".join(reversed(parts))]
    return parts, [m.strip() for m in modifiers if 0 < len(m.split()) <= 3]


_CONNECTORS = {"with", "without", "due to", "in", "of", "and", "see", "by", "following", "complicating", "affecting"}
# Inverted main terms whose qualifier is the default reading of the headword:
# "Fracture, traumatic" is what a plain "scaphoid fracture" means.
_DEFAULT_QUALIFIERS = {"traumatic", "open"}
_UNSPECIFIED_SUBTERMS = ("unspecified site", "unspecified", "site unspecified", "nos")
_NEOPLASM_COLUMNS = (("in situ", "4"), ("secondary", "3"), ("malignant", "2"), ("benign", "5"), ("uncertain", "6"))


def _code_resolver(known_codes: set[str]):
    shortest: dict[str, str] = {}
    for code in sorted(known_codes, key=lambda c: (len(c), c)):
        for length in range(1, len(code) + 1):
            shortest.setdefault(code[:length], code)

    def resolve(raw: str | None) -> str | None:
        if not raw or raw.strip() in {"", "-", "--"}:
            return None
        code = raw.strip().rstrip("-").replace(".", "")
        if len(code) < 3:
            return None
        return code if code in known_codes else shortest.get(code)

    return resolve


def _tree(element, resolve) -> dict[str, Any]:
    """One Alphabetic Index or Neoplasm Table entry with its subterms."""
    title = element.find("title")
    head = re.sub(r"\([^)]*\)", " ", (title.text or "") if title is not None else "")
    parts = [p for p in (_norm(x) for x in head.split(",")) if p]
    variants, modifiers = _index_title_parts(element)
    return {
        "parts": parts,
        "variants": variants,
        "modifiers": modifiers,
        "code": resolve(element.findtext("code")),
        "see": element.findtext("see"),
        "see_also": element.findtext("seeAlso"),
        "cells": {c.get("col"): resolve(c.text) for c in element.findall("cell")},
        "children": [_tree(child, resolve) for child in element.findall("term")],
    }


def _child(node: dict[str, Any], part: str) -> dict[str, Any] | None:
    for child in node["children"]:
        if part in child["parts"] or " ".join(child["parts"]) == part:
            return child
    for child in node["children"]:
        if any(p.startswith(part + " ") or part.startswith(p + " ") for p in child["parts"]):
            return child
    # References often skip a connector level: "Diabetes, retinopathy" is
    # Diabetes > with > retinopathy in the index.
    for connector in node["children"]:
        if connector["parts"] and connector["parts"][0] in _CONNECTORS:
            found = next((c for c in connector["children"] if part in c["parts"]), None)
            if found is not None:
                return found
    return None


def _ref_parts(reference: str) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", reference)
    return [p for p in (_norm(x) for x in text.split(",")) if p and p not in {"by site", "by type", "specified nec"}]


def _load_index_trees(known_codes: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    import xml.etree.ElementTree as ET

    path = REFERENCE_DIR / "icd10cm-April-1-2026-tables-index.zip"
    sha = _download(INDEX_URL, path)
    resolve = _code_resolver(known_codes)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        index_root = ET.fromstring(archive.read(next(x for x in names if re.search(r"icd10cm_index_\d{4}\.xml$", x))))
        neoplasm_root = ET.fromstring(archive.read(next(x for x in names if re.search(r"icd10cm_neoplasm_\d{4}\.xml$", x))))
    mains = [_tree(main, resolve) for main in index_root.iter("mainTerm")]
    neoplasm = _tree(next(neoplasm_root.iter("mainTerm")), resolve)
    return mains, neoplasm, sha


def _neoplasm_site_keys(neoplasm: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Neoplasm Table sites by name: "larynx", "laryngeal", "upper lobe lung"."""
    keys: dict[str, dict[str, Any]] = {}

    def names(node: dict[str, Any]) -> list[str]:
        base = [re.sub(r"\b(nec|nos)\b", " ", p).strip() for p in node["parts"]]
        return [re.sub(r"\s+", " ", b) for b in base if b]

    # A site's own first name wins over a secondary name elsewhere: "lung" is the
    # lung row, not "lingula, lung".
    for site in neoplasm["children"]:
        for name in names(site)[:1]:
            keys.setdefault(name, site)
    for site in neoplasm["children"]:
        for name in names(site):
            keys.setdefault(name, site)
            for modifier in site["modifiers"]:
                keys.setdefault(_norm(f"{name} {modifier}"), site)
                keys.setdefault(_norm(f"{modifier} {name}"), site)
            for sub in site["children"]:
                for sub_name in names(sub):
                    keys.setdefault(f"{sub_name} {name}", sub)
                    keys.setdefault(f"{name} {sub_name}", sub)
    return keys


class _IndexResolver:
    """Follows "see" and "see also" references through the Alphabetic Index,
    including references into the Neoplasm Table ("Neoplasm, bone, malignant")."""

    def __init__(self, mains: list[dict[str, Any]], neoplasm: dict[str, Any]):
        self.neoplasm = neoplasm
        self.by_head: dict[str, list[dict[str, Any]]] = {}
        for main in sorted(mains, key=lambda m: len(m["parts"])):
            if main["parts"]:
                self.by_head.setdefault(main["parts"][0], []).append(main)

    def node_code(self, node: dict[str, Any], depth: int = 0) -> str | None:
        if node["code"]:
            return node["code"]
        if depth > 4:
            return None
        # An "unspecified site" subterm is more specific than a generic
        # "see Neoplasm, malignant, by site" reference on the same term.
        for child in node["children"]:
            if child["parts"] and child["parts"][0] in _UNSPECIFIED_SUBTERMS:
                code = self.node_code(child, depth + 1)
                code = code or next((c for c in (self.node_code(g, depth + 1) for g in child["children"]) if c), None)
                if code:
                    return code
        for reference in (node["see"], node["see_also"]):
            code = self.follow(reference, depth + 1) if reference else None
            if code:
                return code
        return None

    def follow(self, reference: str, depth: int) -> str | None:
        parts = _ref_parts(reference)
        if not parts or parts[0] in {"condition", "code"}:
            return None
        if parts[0].startswith("neoplasm"):
            site = parts[0][len("neoplasm"):].strip()
            return self.neoplasm_code(([site] if site else []) + parts[1:])
        for main in self.by_head.get(parts[0], []):
            node, found = main, True
            for part in [p for p in parts[1:] if p not in main["parts"][1:]]:
                node = _child(node, part)
                if node is None:
                    found = False
                    break
            code = self.node_code(node, depth) if found else None
            if code:
                return code
        return None

    def neoplasm_code(self, parts: list[str]) -> str | None:
        column = None
        sites = []
        for part in parts:
            behavior = next((col for word, col in _NEOPLASM_COLUMNS if word in part), None)
            if behavior and column is None:
                column = behavior
            elif not behavior and part not in {"primary", "neoplastic"}:
                sites.append(part)
        if column is None:
            return None
        node = self.neoplasm
        for site in sites:
            node = _child(node, site)
            if node is None:
                return None
        return node["cells"].get(column)


def load_index(known_codes: set[str], trees: tuple | None = None) -> tuple[dict[str, str], str]:
    """Official ICD-10-CM Alphabetic Index as a term -> code lookup.

    This is how human coders go from a plain clinical name to a code, so it is
    tried before any fuzzy matching. Main terms and their first-level subterms
    are indexed in both word orders ("shock, anaphylactic" and "anaphylactic
    shock"), plus nonessential modifiers ("diabetes (mellitus)"). "See" and
    "see also" references are followed through the index tree at any depth
    ("Graves' disease" -> Hyperthyroidism, with, goiter) and into the Neoplasm
    Table ("Osteosarcoma" -> Neoplasm, bone, malignant). A term with no code of
    its own takes its "unspecified site" subterm. Codes ending in "-" resolve to
    the category code in the order file. Only codes present in the pinned order
    file are kept."""
    mains, neoplasm, sha = trees or _load_index_trees(known_codes)
    resolver = _IndexResolver(mains, neoplasm)
    terms: dict[str, tuple[int, str]] = {}

    def add(term: str, code: str, priority: int) -> None:
        key = _norm(term)
        if key and (key not in terms or priority < terms[key][0]):
            terms[key] = (priority, code)

    for main in mains:
        main_variants, main_mods = main["variants"], main["modifiers"]
        if not main_variants:
            continue
        heads = [main["parts"][0]] if len(main["parts"]) > 1 and main["parts"][1] in _DEFAULT_QUALIFIERS else []
        main_code = resolver.node_code(main)
        referred = 0 if main["code"] else 1
        if main_code:
            for variant in main_variants:
                add(variant, main_code, 0 + referred)
                add(_possessive_free(variant), main_code, 1 + referred)
                for modifier in main_mods:
                    add(f"{variant} {modifier}", main_code, 1 + referred)
                    add(f"{modifier} {variant}", main_code, 1 + referred)
        for sub in main["children"]:
            sub_variants, sub_mods = sub["variants"], sub["modifiers"]
            if not sub_variants or sub_variants[0].casefold() in _CONNECTORS:
                continue
            sub_code = resolver.node_code(sub)
            if sub_code:
                referred = 0 if sub["code"] else 1
                for sub_variant in sub_variants:
                    for variant in main_variants:
                        add(f"{sub_variant} {variant}", sub_code, 2 + referred)
                        add(f"{variant} {sub_variant}", sub_code, 3 + referred)
                        for modifier in sub_mods:
                            add(f"{sub_variant} {modifier} {variant}", sub_code, 3 + referred)
                    for head in heads:
                        add(f"{sub_variant} {head}", sub_code, 6)
                        add(f"{head} of {sub_variant}", sub_code, 7)
            # Second level, e.g. Disorder > depressive > major -> "major depressive disorder".
            for sub2 in sub["children"]:
                sub2_variants = sub2["variants"]
                if not sub2_variants or sub2_variants[0].casefold() in _CONNECTORS:
                    continue
                sub2_code = resolver.node_code(sub2)
                if not sub2_code:
                    continue
                for sub2_variant in sub2_variants:
                    for sub_variant in sub_variants:
                        for variant in main_variants:
                            add(f"{sub2_variant} {sub_variant} {variant}", sub2_code, 4)
                        for head in heads:
                            add(f"{sub2_variant} {sub_variant} {head}", sub2_code, 8)
    return {k: v for k, (_, v) in terms.items()}, sha


def load_neoplasm_sites(known_codes: set[str]) -> dict[str, dict[str, Any]]:
    _, neoplasm, _ = _load_index_trees(known_codes)
    return _neoplasm_site_keys(neoplasm)


def _possessive_free(text: str) -> str:
    """"Kawasaki's syndrome" -> "Kawasaki syndrome", so "Kawasaki disease" can find it."""
    return re.sub(r"(\w{3,})'s\b", r"\1", text)


def _norm(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("'s ", "s ").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_BOILERPLATE = re.compile(
    r"\b(unspecified (organism|site|type|part|cause|laterality)|site unspecified|unspecified|"
    r"not elsewhere classified|without complications?|initial encounter( for closed fracture)?|nos)\b"
)
# Status and encounter codes describe history or health-service contact, not the
# disease itself. They are only chosen when the condition name asks for them.
_CONTEXT_CODE = re.compile(
    r"^(personal history|family history|history of|encounter for|screening for|contact with|exposure to|"
    r"carrier of|long term|observation for|presence of|status|underdosing|other specified|other)\b"
)


def _core(normalized: str) -> str:
    return re.sub(r"\s+", " ", _BOILERPLATE.sub(" ", normalized)).strip()


def format_code(code: str) -> str:
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"
def icd_chapter(code: str) -> str:
    if not code or code == "unmatched":
        return "Unmatched"
    letter = code[0].upper()
    number_match = re.match(r"[A-Z](\d{2})", code.upper())
    number = int(number_match.group(1)) if number_match else -1
    if letter in "AB": return "I. Certain infectious and parasitic diseases"
    if letter == "C" or (letter == "D" and 0 <= number <= 49): return "II. Neoplasms"
    if letter == "D": return "III. Blood and immune disorders"
    if letter == "E": return "IV. Endocrine, nutritional and metabolic diseases"
    if letter == "F": return "V. Mental and behavioral disorders"
    if letter == "G": return "VI. Diseases of the nervous system"
    if letter == "H" and 0 <= number <= 59: return "VII. Diseases of the eye and adnexa"
    if letter == "H": return "VIII. Diseases of the ear and mastoid process"
    mapping = {
        "I": "IX. Diseases of the circulatory system", "J": "X. Diseases of the respiratory system",
        "K": "XI. Diseases of the digestive system", "L": "XII. Diseases of the skin and subcutaneous tissue",
        "M": "XIII. Musculoskeletal and connective tissue diseases", "N": "XIV. Diseases of the genitourinary system",
        "O": "XV. Pregnancy, childbirth and the puerperium", "P": "XVI. Conditions originating in the perinatal period",
        "Q": "XVII. Congenital malformations and chromosomal abnormalities", "R": "XVIII. Symptoms, signs and abnormal findings",
        "S": "XIX. Injury, poisoning and certain other consequences", "T": "XIX. Injury, poisoning and certain other consequences",
        "V": "XX. External causes of morbidity", "W": "XX. External causes of morbidity", "X": "XX. External causes of morbidity",
        "Y": "XX. External causes of morbidity", "Z": "XXI. Factors influencing health status", "U": "XXII. Codes for special purposes",
    }
    return mapping.get(letter, "Unmatched")


def _load_expansions() -> tuple[dict[str, str], dict[str, str]]:
    if not EXPANSIONS_PATH.exists():
        return {}, {}
    data = json.loads(EXPANSIONS_PATH.read_text(encoding="utf-8"))
    names = {str(k): str(v) for k, v in data.get("names", {}).items()}
    words = {_norm(k): _norm(v) for k, v in data.get("words", {}).items()}
    return names, words


# Words that change the reading of a code. A candidate code carrying one of these
# qualifiers loses to a plain code unless the condition name uses matching wording.
_QUALIFIER_RULES = [
    # (applies to candidate, wording in the query that licenses it)
    (lambda code, desc: "D00" <= code[:3] <= "D09" or "in situ" in desc, r"\bin situ\b|\bcis\b"),
    (lambda code, desc: "T80" <= code[:3] <= "T88" or code[:3] in {"E89", "I97", "K91", "N99", "G97", "H59", "H95", "J95", "M96"}
        or re.search(r"\bpost ?(procedural|operative)\b|\bcomplication|\bintraoperative\b|\bfollowing (a )?procedure\b", desc),
        r"\bpost|procedur|surg|operat|device|implant|graft|complicat|transplant|prosthe|iatrogenic"),
    (lambda code, desc: code.startswith("P") or "newborn" in desc or "neonatal" in desc or "perinatal" in desc,
        r"newborn|neonat|perinatal|fetal|fetus|foetal|infant|preterm|premature birth|in utero|birth|maternal"),
    (lambda code, desc: code.startswith("O") or re.search(r"\b(maternal|pregnancy|puerperium|childbirth|gestational)\b", desc),
        r"pregnan|maternal|obstetric|postpartum|puerper|gestation|antepartum|intrapartum|labou?r\b|deliver|fetal|eclampsia|"
        r"cesarean|caesarean|miscarriage|abortion|placent|ectopic|trimester|peripartum|hyperemesis"),
    (lambda code, desc: re.search(r"(?<!non)(?<!non )(?<!a)\btraumatic\b", desc) is not None,
        r"trauma|injur|fracture|contusion|concussion|accident|blow|fall\b|wound"),
    (lambda code, desc: code[:3] in {"E08", "E09"}, r"due to|underlying|drug|chemical|induced|secondary|steroid"),
    (lambda code, desc: "sympathetic" in desc, r"sympathetic"),
    (lambda code, desc: "benign" in desc or "D10" <= code[:3] <= "D36", r"benign|adenoma|papilloma|fibroma|lipoma|hemangioma|haemangioma|nevus|naevus|polyp|myoma|fibroid|"
        r"meningioma|neuroma|schwannoma|osteoma|chondroma|angioma|hamartoma|teratoma|cystadenoma|granular cell tumou?r"),
]
_MALIGNANT_QUERY = re.compile(r"cancer|carcinoma|sarcoma|malignan|lymphoma|leuka?emia|melanoma|blastoma|myeloma|mesothelioma")
_NEOPLASM_QUERY = re.compile(
    r"\b(cancers?|carcinomas?|adenocarcinomas?|malignancy|malignancies|malignant (neoplasm|tumou?r)s?|neoplasm malignant)\b"
)
# Adjectives and lay site names -> Neoplasm Table site names.
_SITE_WORDS = {
    "gastric": "stomach", "colorectal": "colon", "colonic": "colon", "bowel": "colon", "large bowel": "colon",
    "rectal": "rectum", "oral": "mouth", "oral cavity": "mouth", "endometrial": "endometrium", "cervical": "cervix",
    "pancreatic": "pancreas", "hepatic": "liver", "hepatocellular": "liver", "renal": "kidney", "renal cell": "kidney",
    "ovarian": "ovary", "esophageal": "esophagus", "pulmonary": "lung", "mammary": "breast", "vulvar": "vulva",
    "vaginal": "vagina", "testicular": "testis", "penile": "penis", "prostatic": "prostate", "cutaneous": "skin",
    "osseous": "bone", "cerebral": "brain", "duodenal": "duodenum", "appendiceal": "appendix", "thymic": "thymus",
    "tonsillar": "tonsil", "pharyngeal": "pharynx", "oropharyngeal": "oropharynx", "small bowel": "intestine small",
    "small intestine": "intestine small", "urinary bladder": "bladder", "head and neck": "head", "gallbladder": "gallbladder",
    "bile duct": "bile duct", "thyroid gland": "thyroid", "salivary gland": "gland salivary", "adrenal gland": "adrenal",
}
_CANCER_FILLER = re.compile(
    r"\b(invasive|infiltrating|localized|locally|advanced|metastatic|recurrent|primary|early|late|stage|"
    r"non ?small cell|small cell|squamous cell|basal cell|transitional cell|clear cell|cell|ductal|lobular|papillary|"
    r"follicular|medullary|anaplastic|serous|mucinous|urothelial|well|moderately|poorly|differentiated|"
    r"i|ii|iii|iv|of|the|a|nos|unspecified|malignant|left|right|in situ|cis)\b"
)
_SITE_MORPHOLOGY = ("squamous cell carcinoma", "basal cell carcinoma", "melanoma", "merkel cell carcinoma")
_STOP_WORDS = {"of", "the", "a", "an", "and", "or", "in", "with", "to", "joint", "disease", "disorder", "syndrome"}
_STRIPPABLE = re.compile(
    r"^(chronic|acute|arterial|localized|invasive|advanced|severe|mild|moderate|recurrent|early|late|bilateral|"
    r"left|right|suspected|possible|probable|known|untreated|uncontrolled|active|isolated|simple|uncomplicated|"
    r"clinical|mild to moderate|moderate to severe|new onset|symptomatic)\s+"
)


class CodeLookup:
    """Exact match, then the ICD-10-CM Alphabetic Index and Neoplasm Table, then
    rank-aware fuzzy match, then embedding match.

    Names are first expanded with a curated abbreviation list. Ties never fall
    back to file order: candidates are ranked by closeness of the whole name,
    preferring disease codes over history/encounter codes, plain codes over
    qualified ones (in situ, postprocedural, newborn, maternal, traumatic) that
    the name does not ask for, and the most general code among equals."""

    FUZZY_SUBSET_CUTOFF = 90
    FUZZY_ACCEPT = 0.75
    # Between 0.62 and 0.70 most matches on real pilot names were clinically wrong.
    EMBED_ACCEPT = 0.70
    EMBED_MARGIN = 0.03
    RERANK_LIMIT = 10

    def __init__(self, reference: pd.DataFrame, index: dict[str, str] | None = None,
                 neoplasm_sites: dict[str, dict[str, Any]] | None = None):
        self.reference = reference.reset_index(drop=True)
        self.descriptions = self.reference["description"].tolist()
        self.codes = self.reference["code_raw"].tolist()
        self.normalized = [_norm(x) for x in self.descriptions]
        self.cores = [_core(x) for x in self.normalized]
        self.context_code = [bool(_CONTEXT_CODE.search(x)) for x in self.normalized]
        self.exact: dict[str, list[int]] = {}
        self.core_exact: dict[str, list[int]] = {}
        for i, (value, core) in enumerate(zip(self.normalized, self.cores)):
            self.exact.setdefault(value, []).append(i)
            self.core_exact.setdefault(core, []).append(i)
            for synonym in re.findall(r"\[([^\]]+)\]", self.descriptions[i]):
                self.exact.setdefault(_norm(synonym), []).append(i)
        if index is None or neoplasm_sites is None:
            trees = _load_index_trees(set(self.codes))
            if neoplasm_sites is None:
                neoplasm_sites = _neoplasm_site_keys(trees[1])
            if index is None:
                index, _ = load_index(set(self.codes), trees)
        position = {code: i for i, code in enumerate(self.codes)}
        self.position = position
        self.index = {term: position[code] for term, code in index.items() if code in position}
        self.neoplasm_sites = neoplasm_sites
        self.expand_names, self.expand_words = _load_expansions()
        self.expand_casefold = {k.casefold(): v for k, v in self.expand_names.items()}
        self.fuzzy_pool = [i for i, flag in enumerate(self.context_code) if not flag]
        self.fuzzy_cores = [self.cores[i] for i in self.fuzzy_pool]
        self._embedding_model = None
        self._embeddings = None
        self._rerank_cache: dict[str, dict[str, Any]] | None = None

    # ----- query handling -------------------------------------------------
    def variants(self, condition: str) -> list[str]:
        """Normalized readings of a condition name, most literal first."""
        raw = condition.strip()
        forms = [raw]
        if raw.casefold() in self.expand_casefold:
            forms.insert(0, self.expand_casefold[raw.casefold()])
        tokens = raw.split()
        if len(tokens) > 1 and any(t.strip("(),.;:") in self.expand_names and t.strip("(),.;:").isupper() for t in tokens):
            forms.append(" ".join(
                self.expand_names[t.strip("(),.;:")] if t.strip("(),.;:") in self.expand_names and t.strip("(),.;:").isupper() else t
                for t in tokens
            ))
        output: list[str] = []
        for form in forms:
            text = _norm(form)
            spelled = " ".join(self.expand_words.get(w, w) for w in text.split())
            if spelled.casefold() in self.expand_casefold:
                spelled = _norm(self.expand_casefold[spelled.casefold()])
            for value in (text, spelled):
                output.append(value)
                if re.search(r"\b(disease|syndrome)$", value):
                    swapped = re.sub(r"(disease|syndrome)$", lambda m: "syndrome" if m.group(1) == "disease" else "disease", value)
                    output.append(swapped)
                    words = swapped.split()
                    if len(words) >= 2 and not words[-2].endswith("s"):
                        output.append(" ".join(words[:-2] + [words[-2] + "s", words[-1]]))
                last = value.rsplit(" ", 1)[-1]
                if len(last) > 4 and last.endswith("s") and not re.search(r"(ss|is|us|ys|as)$", last):
                    output.append(value[:-1])
        return [x for x in dict.fromkeys(output) if x]

    def _penalty(self, index: int, query: str) -> int:
        code, desc = self.codes[index], self.normalized[index]
        rules = _QUALIFIER_RULES
        if self._plain_diabetes(index, query) != index:
            # E08/E09 is swapped for its plain E11/E10 twin, so it is not a wrong reading.
            rules = [r for r in rules if not r[0](code, desc) or code[:3] not in {"E08", "E09"}]
        penalty = sum(1 for applies, licensed in rules if applies(code, desc) and not re.search(licensed, query))
        if _MALIGNANT_QUERY.search(query) and not code.startswith("C") and not _MALIGNANT_QUERY.search(desc):
            penalty += 1
        return penalty

    def _missing_words(self, index: int, query_core: str) -> set[str]:
        """Content words of the name that the candidate description lacks
        ("posterior dislocation of knee joint" vs an ulnohumeral code lacks "knee").
        A misspelled word ("ahogryposis") counts as present when it is close to one."""
        described = set(self.cores[index].split())
        missing = set(query_core.split()) - described - _STOP_WORDS

        def close(word: str, other: str) -> bool:
            # Misspellings ("ahogryposis") and word forms ("uterine"/"uterus", "diabetic"/"diabetes").
            return len(word) > 4 and (fuzz.ratio(word, other) >= 80 or (len(other) > 4 and word[:5] == other[:5]))

        return {w for w in missing if not any(close(w, d) for d in described)}

    def _other_party(self, index: int, query: str) -> bool:
        """Codes describing a different person than the patient named: the newborn in
        "newborn affected by maternal ...", the mother in "maternal care for fetal ..."."""
        desc = self.normalized[index]
        if re.match(r"(newborn|fetus) (suspected to be )?affected by", desc):
            return not re.search(r"newborn|neonat|fetal|fetus|foetal|perinatal|in utero|infant", query)
        if desc.startswith("maternal care for"):
            return not re.search(r"pregnan|maternal|antenatal|prenatal|obstetric", query)
        return False

    def _plain_diabetes(self, index: int, query: str) -> int:
        """E08/E09 ("due to underlying condition", "drug induced") -> the same
        complication under E11 (or E10 for type 1), when the name does not ask
        for a secondary cause. Only codes present in the official file are used."""
        code = self.codes[index]
        if code[:3] not in {"E08", "E09"} or re.search(r"due to|underlying|drug|chemical|induced|secondary|steroid", query):
            return index
        family = "E10" if re.search(r"\btype (1|i)\b|juvenile|insulin dependent", query) else "E11"
        return self.position.get(family + code[3:], index)

    def _rank_key(self, index: int, query_core: str) -> tuple:
        return (
            self.context_code[index],
            self._penalty(index, query_core),
            len(self._missing_words(index, query_core)),
            -fuzz.ratio(query_core, self.cores[index]),
            len(self.codes[index]),
            self.codes[index],
        )

    def _best(self, indices: list[int], query_core: str, allow_context: bool) -> int | None:
        pool = indices if allow_context else [i for i in indices if not self.context_code[i]]
        pool = pool or indices
        return min(pool, key=lambda i: self._rank_key(i, query_core)) if pool else None

    def _neoplasm_match(self, query: str) -> int | None:
        """"laryngeal carcinoma" -> Neoplasm Table, larynx, malignant primary."""
        if not _NEOPLASM_QUERY.search(query):
            return None
        column = "4" if re.search(r"\bin situ\b|\bcis\b", query) else "2"
        morphology = next((m for m in _SITE_MORPHOLOGY if m in query), None)
        site = _NEOPLASM_QUERY.sub(" ", query)
        for phrase, noun in sorted(_SITE_WORDS.items(), key=lambda kv: -len(kv[0])):
            site = re.sub(rf"\b{phrase}\b", noun, site)
        site = re.sub(r"\s+", " ", _CANCER_FILLER.sub(" ", site)).strip()
        node = self.neoplasm_sites.get(site) if site else None
        if node is None:
            return None
        code = node["cells"].get(column)
        if morphology:
            sub = _child(node, morphology)
            code = (sub["cells"].get(column) if sub else None) or code
        return self.position.get(code) if code else None

    def _deterministic(self, query: str, query_core: str, allow_context: bool) -> tuple[int, str] | None:
        if query in self.exact:
            return self._best(self.exact[query], query_core, allow_context), "exact"
        for key in dict.fromkeys((query, query_core)):
            if key in self.index:
                return self.index[key], "icd_index"
        if query_core in self.core_exact:
            return self._best(self.core_exact[query_core], query_core, allow_context), "exact"
        found = self._neoplasm_match(query)
        if found is not None:
            return found, "neoplasm_table"
        return None

    def _fuzzy(self, query_core: str, allow_context: bool, cutoff: int | None = None, limit: int = 300) -> list[int]:
        candidates = process.extract(
            query_core, self.fuzzy_cores, scorer=fuzz.token_set_ratio,
            score_cutoff=cutoff or self.FUZZY_SUBSET_CUTOFF, limit=limit,
        )
        pool = [self.fuzzy_pool[c[2]] for c in candidates]
        return sorted(pool, key=lambda i: self._rank_key(i, query_core))

    def _embedding_top(self, condition: str, limit: int = 20) -> list[tuple[int, float]]:
        self._load_embeddings()
        vector = self._embedding_model.encode([condition], normalize_embeddings=True)[0]
        scores = self._embeddings @ vector
        top = np.argsort(scores)[::-1][:limit]
        return [(self.fuzzy_pool[int(i)], float(scores[i])) for i in top]

    def _row(self, index: int, method: str, score: float) -> dict[str, Any]:
        ref = self.reference.iloc[index]
        raw_code = str(ref.code_raw)
        return {
            "code": format_code(raw_code),
            "matched_description": str(ref.description),
            "match_method": method,
            "match_score": round(float(score), 4),
            "ccsr_category": str(ref.ccsr_category),
            "ccsr_description": str(ref.ccsr_description),
            "icd_chapter": icd_chapter(raw_code),
            "icd_version": ICD_VERSION,
            "ccsr_version": CCSR_VERSION,
        }

    def _load_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._embedding_model = SentenceTransformer(model_name)
        texts = [self.descriptions[i] for i in self.fuzzy_pool]
        cache = REFERENCE_DIR / "icd_description_embeddings.npy"
        fingerprint = REFERENCE_DIR / "icd_description_embeddings.sha256"
        expected = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        if cache.exists() and fingerprint.exists() and fingerprint.read_text().strip() == expected:
            self._embeddings = np.load(cache)
        else:
            print(f"Encoding {len(texts):,} ICD descriptions once for the embedding fallback.")
            self._embeddings = self._embedding_model.encode(
                texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
            )
            np.save(cache, self._embeddings)
            fingerprint.write_text(expected, encoding="utf-8")

    def match(self, condition: str) -> dict[str, Any]:
        variants = self.variants(condition)
        if not variants:
            return self.unmatched()
        query = variants[0]
        allow_context = bool(_CONTEXT_CODE.search(query))
        # Literal readings first, then the same readings without a leading
        # nonessential adjective ("chronic hypertension" -> hypertension).
        stripped = [s for s in (_STRIPPABLE.sub("", v) for v in variants) if s not in variants]
        for value in variants + stripped:
            found = self._deterministic(value, _core(value) or value, allow_context)
            if found is not None and found[0] is not None:
                return self._row(self._plain_diabetes(found[0], query), found[1], 1.0)
        # In the fuzzy step, qualified codes (newborn, maternal, postprocedural ...) rank
        # below plain ones but are not discarded: "meconium aspiration" has only newborn
        # codes. Every content word of the name must still appear in the description.
        for value in variants:
            query_core = _core(value) or value
            pool = self._fuzzy(query_core, allow_context)
            pool = [i for i in pool if allow_context or not self.context_code[i]] or pool
            if pool and not self._missing_words(pool[0], query_core):
                score = fuzz.ratio(query_core, self.cores[pool[0]]) / 100
                if score >= self.FUZZY_ACCEPT:
                    return self._row(self._plain_diabetes(pool[0], query), "fuzzy", score)
        query_core = _core(query) or query
        top = self._embedding_top(condition)
        best_score = top[0][1]
        near = [(i, s) for i, s in top if s >= self.EMBED_ACCEPT and best_score - s < self.EMBED_MARGIN]
        # The embedding step is the loosest. A qualified code is kept only when every
        # word of the name is in its description ("malignant hyperthermia" -> "...due to
        # anesthesia"), and never when the code is about someone else ("cesarean
        # delivery" -> "newborn affected by cesarean delivery").
        near = [
            (i, s) for i, s in near
            if self._penalty(i, query_core) == 0
            or (not self._missing_words(i, query_core) and not self._other_party(i, query_core))
        ]
        if near:
            lowest = min(self._penalty(i, query_core) for i, _ in near)
            near = [(i, s) for i, s in near if self._penalty(i, query_core) == lowest]
        if near:
            ties = [i for i, s in near if near[0][1] - s < 0.01]
            best = self._best(ties, query_core, allow_context)
            return self._row(self._plain_diabetes(best, query), "embedding", dict(near)[best])
        return self.unmatched(best_score)

    # ----- candidate rerank ----------------------------------------------
    def candidates(self, condition: str, base: dict[str, Any] | None = None) -> list[int]:
        """Up to RERANK_LIMIT distinct real codes worth showing a model."""
        variants = self.variants(condition)
        if not variants:
            return []
        query = variants[0]
        query_core = _core(query) or query
        found: list[int] = []
        if base and base.get("code") not in (None, "unmatched"):
            found.append(self.position[base["code"].replace(".", "")])
        for value in variants:
            core = _core(value) or value
            found += self.exact.get(value, []) + self.core_exact.get(core, [])
            found += [self.index[k] for k in (value, core) if k in self.index]
            hit = self._neoplasm_match(value)
            found += [hit] if hit is not None else []
        found += self._fuzzy(query_core, True, cutoff=80, limit=100)[:5]
        found += [i for i, s in self._embedding_top(condition, 8) if s >= 0.5]
        unique = [i for i in dict.fromkeys(found) if not self.context_code[i] or bool(_CONTEXT_CODE.search(query))]
        unique.sort(key=lambda i: self._penalty(i, query_core))
        return unique[: self.RERANK_LIMIT]

    def rerank(self, condition: str, client: Any, base: dict[str, Any] | None = None,
               checkpoint: Path = RERANK_CHECKPOINT) -> dict[str, Any]:
        """Ask a model to choose among real candidate codes, or none. The model
        never writes a code: its answer is an index into the candidate list."""
        base = base or self.match(condition)
        shown = self.candidates(condition, base)
        if not shown:
            return base
        listed = [{"number": n, "code": format_code(self.codes[i]), "description": self.descriptions[i]} for n, i in enumerate(shown)]
        candidate_hash = sha256_text(json.dumps([c["code"] for c in listed]))
        key = sha256_text(json.dumps([condition.strip(), LOOKUP_VERSION, candidate_hash]))
        if self._rerank_cache is None:
            self._rerank_cache = {x["key"]: x for x in read_jsonl(checkpoint) if "key" in x}
        cached = self._rerank_cache.get(key)
        if cached is None:
            template = RERANK_PROMPT.read_text(encoding="utf-8")
            prompt = template.replace("{{CONDITION}}", condition.strip()).replace(
                "{{CANDIDATES}}", "\n".join(f"{c['number']}. {c['code']} {c['description']}" for c in listed)
            )
            error, choice, reason = "", None, ""
            try:
                response = client.generate_json(prompt)
                raw = response.get("choice")
                reason = str(response.get("reason", ""))[:300]
                if isinstance(raw, str) and raw.strip().casefold() == "none":
                    choice = "none"
                elif isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.strip().isdigit()):
                    choice = int(raw) if 0 <= int(raw) < len(listed) else None
                    error = "" if choice is not None else f"choice out of range: {raw!r}"
                else:
                    error = f"unreadable choice: {raw!r}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            cached = {
                "key": key, "condition_name": condition.strip(), "lookup_version": LOOKUP_VERSION,
                "candidate_hash": candidate_hash, "candidates": listed, "choice": choice,
                "chosen_code": listed[choice]["code"] if isinstance(choice, int) else "", "reason": reason, "error": error,
                "model_name": getattr(client, "model_name", "unknown"),
                "model_settings": json.dumps(getattr(client, "settings", {}), sort_keys=True),
                "prompt_version": prompt_version(RERANK_PROMPT), "prompt_sha256": sha256_text(template), "created_at": now_iso(),
            }
            if not error:
                append_jsonl(checkpoint, [cached])
                self._rerank_cache[key] = cached
        if cached.get("error") or cached.get("choice") is None:
            return base
        if cached["choice"] == "none":
            return {**self.unmatched(), "match_method": "rerank_none"}
        index = shown[int(cached["choice"])]
        same = format_code(self.codes[index]) == base.get("code")
        score = base["match_score"] if same else fuzz.ratio(_core(_norm(condition)), self.cores[index]) / 100
        return self._row(index, "rerank", score)

    @staticmethod
    def unmatched(score: float = 0.0) -> dict[str, Any]:
        return {
            "code": "unmatched", "matched_description": "", "match_method": "unmatched",
            "match_score": round(score, 4), "ccsr_category": "unmatched", "ccsr_description": "Unmatched",
            "icd_chapter": "Unmatched", "icd_version": ICD_VERSION, "ccsr_version": CCSR_VERSION,
        }


def run() -> pd.DataFrame:
    ensure_dirs()
    labels_path = DATA / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError("Stage 2 output data/labels.parquet is missing.")
    labels = pd.read_parquet(labels_path)
    reference, manifest = load_reference()
    lookup = CodeLookup(reference)
    # Optional model rerank over real candidate codes. Rerank decisions change the
    # mapping, so they are cached under their own lookup version.
    rerank_client = None
    version = LOOKUP_VERSION
    if os.getenv("CODE_RERANK") == "1":
        from .models import choose_primary_model

        rerank_client, _ = choose_primary_model(allow_pull=False)
        version = f"{LOOKUP_VERSION}+rerank-{prompt_version(RERANK_PROMPT)}-{rerank_client.model_name}"
    cached_rows = [
        x for x in read_jsonl(MAPPING_CHECKPOINT)
        if x.get("icd_version") == ICD_VERSION and x.get("ccsr_version") == CCSR_VERSION
        and x.get("lookup_version") == version
    ]
    cached = {str(x["condition_name"]).casefold(): x for x in cached_rows}
    unique_conditions = sorted({str(x).strip() for x in labels["condition_name"] if str(x).strip()})
    for index, condition in enumerate(unique_conditions, start=1):
        key = condition.casefold()
        if key in cached:
            continue
        result = lookup.match(condition)
        reranked = rerank_client is not None and result["match_method"] in RERANK_METHODS
        if reranked:
            result = lookup.rerank(condition, rerank_client, base=result)
        mapping = {
            "condition_name": condition,
            **result,
            "lookup_version": version,
            "prompt_text": f"prompts/code_rerank.md version {prompt_version(RERANK_PROMPT)}" if reranked else "",
            "model_name": rerank_client.model_name if reranked else "none",
            "model_settings": json.dumps(rerank_client.settings, sort_keys=True) if reranked else "{}",
        }
        append_jsonl(MAPPING_CHECKPOINT, [mapping])
        cached[key] = mapping
        if index % 100 == 0:
            print(f"Coded {index:,} of {len(unique_conditions):,} unique condition names.")

    coding_fields = ["code", "matched_description", "match_method", "match_score", "ccsr_category", "ccsr_description", "icd_chapter", "icd_version", "ccsr_version"]
    output_rows = []
    for row in labels.to_dict("records"):
        mapping = cached.get(str(row["condition_name"]).casefold(), CodeLookup.unmatched())
        output_rows.append({**row, **{field: mapping[field] for field in coding_fields}})
    coded = pd.DataFrame(output_rows)
    write_parquet_atomic(coded, DATA / "coded.parquet")
    report = (
        coded.assign(unmatched=coded["code"].eq("unmatched") & coded["condition_name"].ne(""))
        .groupby("source", dropna=False)
        .agg(condition_rows=("condition_name", lambda x: int((x != "").sum())), unmatched=("unmatched", "sum"))
        .reset_index()
    )
    report["unmatched_rate"] = report.apply(lambda x: x.unmatched / x.condition_rows if x.condition_rows else 0.0, axis=1)
    text = (
        "# Coding report\n\n"
        f"Conditions were matched to official ICD-10-CM descriptions and rolled up with AHRQ CCSR {CCSR_VERSION}. "
        "The pipeline expanded curated abbreviations (prompts/condition_abbreviations.json), then used exact matching against code descriptions, then the official ICD-10-CM Alphabetic Index with its see-references and Neoplasm Table, then fuzzy matching, then an embedding nearest-neighbor match. Qualified codes (in situ, postprocedural, newborn, maternal, traumatic) lose to plain codes unless the name asks for them. "
        + ("Fuzzy, embedding, and unmatched names were then reranked by a local model choosing only among real candidate codes (prompts/code_rerank.md). " if rerank_client else "")
        + "Ties are ranked by closeness of the whole name, history and encounter codes are chosen only when the condition name asks for them, and category-level codes roll up to the most common CCSR category of their billable codes. "
        "Embedding matches below 0.70 remain unmatched; the pipeline never fabricates a code.\n\n"
        + report.to_markdown(index=False, floatfmt=".3f")
        + "\n\n"
        + f"Pinned ICD file SHA-256: `{manifest['icd_sha256']}`. Pinned CCSR file SHA-256: `{manifest['ccsr_sha256']}`.\n"
    )
    named = coded[coded["condition_name"] != ""]
    methods = named["match_method"].value_counts().rename_axis("match_method").rename("condition_rows").reset_index()
    text += "\nHow condition rows were matched:\n\n" + methods.to_markdown(index=False) + "\n"
    (DATA / "coding_report.md").write_text(text, encoding="utf-8")
    append_log("Stage 3 coding", f"Unique conditions: {len(unique_conditions):,}  \nUnmatched: {int((coded['code'] == 'unmatched').sum()):,}  \nCCSR version: `{CCSR_VERSION}`")
    print(f"Stage 3 complete: wrote {len(coded):,} rows to data/coded.parquet.")
    return coded
