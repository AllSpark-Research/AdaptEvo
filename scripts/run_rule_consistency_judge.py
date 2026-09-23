#!/usr/bin/env python3
"""Batch rule/evidence consistency judging with an OpenAI-compatible model.

The judge GT is always taken from the xlsx_correct_label round in
metadata.four_round_labels.  Each request contains the full post, post images,
source-specific rules, and all nine cached audit-tool observations (including
their rendered images).

Use --dry-run to validate data/rule/tool/image preparation without an endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


REPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_PROJECT_ROOT = REPORT_ROOT
DEFAULT_PROMPT = REPORT_ROOT / "prompt_templates/prompt.jinja2"
DEFAULT_RULE_CACHE = REPORT_ROOT / "local_artifacts/rules_cache.jsonl"
DEFAULT_TOOL_CACHE = REPORT_ROOT / "local_artifacts/tool_cache"
DEFAULT_TOOL_CACHE_VERSION = "local-v1"
REQUIRED_TOOL_NAMES = (
    "check_plagiarism_video",
    "check_plagiarism_image",
    "get_commercial_detail",
    "get_user_qualification",
    "get_recent_notes",
    "get_comments",
    "get_user_records",
    "get_report_history",
    "get_user_comments",
)
DEFAULT_CACHE_ENV = {
    "BLOB_INDEX_PATH": "",
    "IMAGE_CACHE_DIR": "",
    "PLAGIARISM_CACHE_PATH": "",
    "AUDIT_TOOL_DATA_PATH": "",
    "RECENT_NOTES_MAX_NOTES": "",
    "RECENT_NOTES_MAX_IMAGES_PER_NOTE": "",
    "COMMERCIAL_DETAIL_MAX_IMAGES_PER_ITEM": "",
}
SYSTEM_PROMPT = (
    "你是一名内容审核规则-证据一致性 Judge。请严格依据用户消息中的帖子、"
    "工具结果和规则完成核验，并且只输出要求的 JSON 对象。"
)
PASS_ISSUE_TYPES = {
    "none",
    "rule_ambiguity",
    "missing_information",
    "scale_dependency",
    "violation_supported",
}
VIOLATION_ISSUE_TYPES = {
    "none",
    "rule_ambiguity",
    "missing_information",
    "scale_dependency",
    "rule_conflict",
}
VERDICTS = {"supported", "ambiguous", "unsupported"}
IMAGE_TAG = "<image>"
_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a strong multimodal model to judge whether xlsx_correct_label "
            "is supported by the post, nine tool caches, and source rules."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Clean Train/Test JSONL")
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--rule-cache", type=Path, default=DEFAULT_RULE_CACHE)
    parser.add_argument("--tool-cache", type=Path, default=DEFAULT_TOOL_CACHE)
    parser.add_argument("--tool-cache-version", default=DEFAULT_TOOL_CACHE_VERSION)
    parser.add_argument("--audit-project-root", type=Path, default=DEFAULT_AUDIT_PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory; defaults to report/outputs/rule_consistency_judge/<input>-<time>",
    )

    parser.add_argument("--base-url", "--url", dest="base_url")
    parser.add_argument("--model")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SERVE_API_KEY"),
        help="Optional API key; defaults to SERVE_API_KEY from the environment",
    )
    parser.add_argument(
        "--api-key-header",
        default="Authorization",
        help="Header used by --api-key; Authorization values receive a Bearer prefix",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional request header; repeat for multiple headers",
    )
    parser.add_argument(
        "--extra-body-json",
        help="Optional JSON object merged into the request body (inline JSON or a file path)",
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP read timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument(
        "--omit-thinking-kwargs",
        action="store_true",
        help="Do not send chat_template_kwargs.enable_thinking",
    )
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)

    parser.add_argument("--image-max-tokens", type=int, default=448)
    parser.add_argument(
        "--image-transport",
        choices=["base64", "file"],
        default="base64",
        help="base64 is required when the endpoint cannot access shared local paths",
    )
    parser.add_argument(
        "--image-base64-max-bytes",
        type=int,
        default=32 * 1024 * 1024,
        help="Per-image cap after validation/resize",
    )

    parser.add_argument("--limit", type=int, help="Process at most N selected rows")
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Only process these source names (space-separated)",
    )
    parser.add_argument(
        "--row-indices",
        help="Only process zero-based row indices, e.g. 0,3,8-12",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Deterministically split selected rows by row_idx modulo this value",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to process when --num-shards is greater than one",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, rerun rows whose latest result was unsuccessful",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare requests without HTTP calls")
    parser.add_argument("--save-prompt", action="store_true", help="Store full rendered prompt per row")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalize_endpoint(base_url: str) -> str:
    url = str(base_url or "").strip().rstrip("/,， ")
    if not url:
        raise ValueError("--base-url/--url is required unless --dry-run is used")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def parse_headers(values: list[str], api_key: Optional[str], api_key_header: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        value = api_key
        if api_key_header.lower() == "authorization" and not value.lower().startswith("bearer "):
            value = f"Bearer {value}"
        headers[api_key_header] = value
    for item in values:
        if "=" in item:
            name, value = item.split("=", 1)
        elif ":" in item:
            name, value = item.split(":", 1)
        else:
            raise ValueError(f"Invalid --header {item!r}; expected NAME=VALUE")
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --header {item!r}; header name is empty")
        headers[name] = value.strip()
    return headers


def load_extra_body(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    raw = path.read_text(encoding="utf-8") if path.is_file() else value
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    return payload


def parse_row_indices(spec: Optional[str]) -> Optional[set[int]]:
    if not spec:
        return None
    selected: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid row-index range: {token}")
            selected.update(range(start, end + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError(f"Invalid negative row index: {index}")
            selected.add(index)
    return selected


def strip_label_id(value: Any) -> str:
    label = str(value or "").strip()
    return label.split("|", 1)[1].strip() if "|" in label else label


def label_compare_key(value: Any) -> str:
    """Normalize harmless typography differences without changing label semantics."""
    label = unicodedata.normalize("NFKC", strip_label_id(value))
    label = re.sub(r"[‐‑‒–—−]", "-", label)
    return re.sub(r"\s+", "", label)


def canonical_output_label(value: Any, candidates: list[str]) -> Any:
    if value is None:
        return None
    key = label_compare_key(value)
    matches = [label for label in candidates if label_compare_key(label) == key]
    return matches[0] if len(matches) == 1 else value


def normalized_label_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    labels: list[str] = []
    for item in values:
        label = strip_label_id(item)
        if label and label not in labels:
            labels.append(label)
    return labels


def extract_xlsx_gt(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    four_round = metadata.get("four_round_labels")
    if not isinstance(four_round, dict):
        return []
    rounds = four_round.get("rounds")
    if not isinstance(rounds, list):
        return []
    for entry in rounds:
        if isinstance(entry, dict) and entry.get("name") == "xlsx_correct_label":
            return normalized_label_list(entry.get("labels"))
    return []


def load_rules(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source") or "").strip()
            label = strip_label_id(record.get("label"))
            if not source or not label:
                raise ValueError(f"Invalid rule cache row {line_no}: source/label missing")
            normalized = dict(record)
            normalized["source"] = source
            normalized["label"] = label
            normalized["rule_text"] = str(record.get("rule_text") or "")
            normalized["rule_images"] = [str(v) for v in (record.get("rule_images") or []) if v]
            by_source[source].append(normalized)
    return dict(by_source)


def cache_key(version: str, tool_name: str, note_id: str, history_id: str) -> str:
    payload = {
        "version": version,
        "tool_name": tool_name,
        "args": {"note_id": note_id, "history_id": history_id},
        "env": DEFAULT_CACHE_ENV,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_path(root: Path, tool_name: str, key: str) -> Path:
    return root / tool_name / key[:2] / key[2:4] / f"{key}.json"


def align_text_images(text: str, images: Iterable[str]) -> tuple[str, list[str]]:
    text = str(text or "")
    image_list = [str(value) for value in images if value]
    tag_count = text.count(IMAGE_TAG)
    if tag_count > len(image_list):
        parts = text.split(IMAGE_TAG)
        rebuilt = parts[0]
        for index, part in enumerate(parts[1:]):
            rebuilt += (IMAGE_TAG if index < len(image_list) else "图片") + part
        text = rebuilt
    elif len(image_list) > tag_count:
        image_list = image_list[:tag_count]
    return text, image_list


def select_case_rules(
    source_rules: list[dict[str, Any]],
    candidate_labels: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    candidate_set = set(candidate_labels)
    selected: list[dict[str, Any]] = []
    found: set[str] = set()
    for rule in source_rules:
        label = rule["label"]
        if label not in candidate_set:
            continue
        normalized = dict(rule)
        text, images = align_text_images(rule.get("rule_text", ""), rule.get("rule_images", []))
        normalized["rule_text"] = text
        normalized["rule_images"] = images
        selected.append(normalized)
        found.add(label)
    missing = [label for label in candidate_labels if label != "通过" and label not in found]
    return selected, missing


def rule_images_in_prompt_order(
    rules: list[dict[str, Any]],
    gt_labels: list[str],
) -> list[str]:
    images: list[str] = []
    is_pass_gt = gt_labels == ["通过"]
    target_labels = set(gt_labels)
    for rule in rules:
        label = rule.get("label")
        included = (
            label not in {"通过", "__queue_notice__"}
            if is_pass_gt
            else label in target_labels
        )
        if included:
            images.extend(str(value) for value in (rule.get("rule_images") or []) if value)
    return images


def render_tool_section(
    note_id: str,
    history_id: str,
    registry: dict[str, Any],
    cache_root: Path,
    cache_version: str,
) -> tuple[str, list[str], list[str], list[dict[str, str]]]:
    blocks: list[str] = []
    all_images: list[str] = []
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    for name, tool in registry.items():
        key = cache_key(cache_version, name, note_id, history_id)
        path = cache_path(cache_root, name, key)
        if not path.is_file():
            missing.append(name)
            blocks.append(
                f"## 工具：{name}\n- cache_status: missing\n- result:\n（未找到该工具缓存。）"
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key") not in {None, key}:
                raise ValueError("cache key mismatch")
            if "result" not in payload:
                raise ValueError("cache payload has no result field")
            rendered = tool.render(payload.get("result"))
            text, images = align_text_images(rendered.text, rendered.images)
            blocks.append(f"## 工具：{name}\n- cache_status: ok\n- result:\n{text}")
            all_images.extend(images)
        except Exception as exc:
            errors.append({"tool": name, "error": f"{type(exc).__name__}: {exc}", "path": str(path)})
            blocks.append(
                f"## 工具：{name}\n- cache_status: error\n- result:\n"
                f"（工具缓存读取或渲染失败：{type(exc).__name__}。）"
            )
    return "\n\n".join(blocks).strip(), all_images, missing, errors


def load_selected_rows(
    path: Path,
    sources: Optional[set[str]],
    row_indices: Optional[set[int]],
    limit: Optional[int],
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if row_indices is not None and row_idx not in row_indices:
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            source = str(metadata.get("source") or (metadata.get("audit_input") or {}).get("source_id") or "")
            if sources is not None and source not in sources:
                continue
            selected.append((row_idx, row))
            if limit is not None and len(selected) >= limit:
                break
    return selected


def load_latest_results(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                latest[int(row["row_idx"])] = row
            except Exception:
                continue
    return latest


def should_skip_existing(existing: Optional[dict[str, Any]], retry_failed: bool) -> bool:
    if existing is None:
        return False
    if not retry_failed:
        return True
    return bool(existing.get("success"))


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "".join(chunks)
    return "" if content is None else str(content)


def balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def parse_output_json(content: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    stripped = str(content or "").strip()
    if not stripped:
        return None, "assistant content is empty"
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "top-level JSON is not an object"
    except json.JSONDecodeError:
        pass

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = balanced_json_objects(fenced.group(1) if fenced else stripped)
    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            valid.append(parsed)
    if len(valid) == 1:
        return valid[0], None
    if not valid:
        return None, "no unambiguous JSON object found"
    return None, f"multiple JSON objects found ({len(valid)})"


def validate_judge_output(
    parsed: Optional[dict[str, Any]],
    gt_labels: list[str],
    candidate_labels: list[str],
) -> list[str]:
    if parsed is None:
        return ["parsed output is missing"]
    errors: list[str] = []
    verdict = parsed.get("verdict")
    issue_type = parsed.get("issue_type")
    key_rule_label = parsed.get("key_rule_label")
    brief_reason = parsed.get("brief_reason")
    is_pass_gt = gt_labels == ["通过"]
    if verdict not in VERDICTS:
        errors.append(f"invalid verdict: {verdict!r}")
    allowed_issue_types = PASS_ISSUE_TYPES if is_pass_gt else VIOLATION_ISSUE_TYPES
    if issue_type not in allowed_issue_types:
        errors.append(f"invalid issue_type for GT branch: {issue_type!r}")
    if verdict == "supported" and issue_type != "none":
        errors.append("supported must use issue_type=none")
    if verdict != "supported" and issue_type == "none":
        errors.append("ambiguous/unsupported cannot use issue_type=none")
    if is_pass_gt:
        violation_candidates = {label for label in candidate_labels if label != "通过"}
        normalized_key_rule_label = canonical_output_label(key_rule_label, list(violation_candidates))
        if key_rule_label is not None and normalized_key_rule_label not in violation_candidates:
            errors.append(f"pass-GT key_rule_label is not a candidate violation label: {key_rule_label!r}")
        if verdict == "unsupported" and issue_type != "violation_supported":
            errors.append("pass-GT unsupported must use issue_type=violation_supported")
    else:
        normalized_key_rule_label = canonical_output_label(key_rule_label, gt_labels)
        if normalized_key_rule_label not in gt_labels:
            errors.append(
                "violation-GT key_rule_label must be one of the xlsx GT labels "
                f"{gt_labels!r}, got {key_rule_label!r}"
            )
    if not isinstance(brief_reason, str) or not brief_reason.strip():
        errors.append("brief_reason must be a non-empty string")
    return errors


def get_http_session() -> Any:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        import requests

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def call_model(
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any],
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
) -> tuple[dict[str, Any], int]:
    retry_statuses = {408, 409, 429, 500, 502, 503, 504}
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = get_http_session().post(
                endpoint,
                headers=headers,
                json=body,
                timeout=(connect_timeout, read_timeout),
            )
            if response.status_code in retry_statuses and attempt < max_retries:
                delay = min(15.0, (2**attempt) + random.random())
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                preview = response.text[:2000]
                raise RuntimeError(f"HTTP {response.status_code}: {preview}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("API response is not a JSON object")
            return payload, attempt
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(15.0, (2**attempt) + random.random()))
    assert last_error is not None
    raise last_error


def make_base_result(row_idx: int, audit_input: Any, gt_labels: list[str]) -> dict[str, Any]:
    is_pass_gt = gt_labels == ["通过"]
    return {
        "row_idx": row_idx,
        "note_id": str(audit_input.note_id or ""),
        "history_id": str(audit_input.history_id or ""),
        "source": str(audit_input.source_id or ""),
        "candidate_labels": list(audit_input.candidate_labels or []),
        "gt_labels": gt_labels,
        "gt_type": "pass" if is_pass_gt else "violation" if gt_labels else None,
        "success": False,
        "http_success": False,
        "dry_run": False,
        "error": None,
        "error_type": None,
        "verdict": None,
        "issue_type": None,
        "key_rule_label": None,
        "key_rule_label_raw": None,
        "brief_reason": None,
        "raw_response": "",
        "reasoning_content": "",
        "strict_parse_valid": False,
        "strict_parse_error": None,
        "strict_validation_errors": [],
        "output_warnings": [],
        "latency_seconds": None,
        "retry_count": 0,
        "usage": {},
        "finish_reason": None,
        "post_image_count": len(audit_input.images or []),
        "tool_image_count": 0,
        "rule_image_count": 0,
        "final_image_count": 0,
        "dropped_images": [],
        "converted_images": [],
        "tool_cache_missing": [],
        "tool_cache_errors": [],
        "rule_missing": [],
        "rendered_prompt_chars": 0,
        "rendered_prompt_sha256": None,
    }


def process_case(
    row_idx: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    template: Any,
    rules_by_source: dict[str, list[dict[str, Any]]],
    registry: dict[str, Any],
    modules: dict[str, Any],
    endpoint: Optional[str],
    headers: dict[str, str],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        audit_input = modules["AuditInput"].from_data_row(row)
    except Exception as exc:
        return {
            "row_idx": row_idx,
            "success": False,
            "error_type": "input_parse_error",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": time.monotonic() - started,
        }

    gt_labels = extract_xlsx_gt(row)
    result = make_base_result(row_idx, audit_input, gt_labels)
    try:
        if not gt_labels:
            raise ValueError("xlsx_correct_label is empty")
        if "通过" in gt_labels and gt_labels != ["通过"]:
            raise ValueError(f"pass cannot coexist with violation GT labels: {gt_labels!r}")
        candidate_labels = list(audit_input.candidate_labels or [])
        missing_gt_candidates = [label for label in gt_labels if label not in candidate_labels]
        if missing_gt_candidates:
            raise ValueError(
                "xlsx GT labels are absent from candidate_labels: "
                f"{missing_gt_candidates!r}"
            )
        if not audit_input.note_id or not audit_input.history_id:
            raise ValueError("note_id/history_id is missing")
        source = str(audit_input.source_id or "")
        if source not in rules_by_source:
            raise ValueError(f"source {source!r} is absent from the rule cache")

        case_rules, rule_missing = select_case_rules(rules_by_source[source], candidate_labels)
        result["rule_missing"] = rule_missing
        tool_section, tool_images, tool_missing, tool_errors = render_tool_section(
            str(audit_input.note_id),
            str(audit_input.history_id),
            registry,
            args.tool_cache,
            args.tool_cache_version,
        )
        result["tool_cache_missing"] = tool_missing
        result["tool_cache_errors"] = tool_errors
        result["tool_image_count"] = len(tool_images)

        rendered_prompt = template.render(
            note=audit_input.note,
            candidate_labels=candidate_labels,
            gt_labels=gt_labels,
            rules=case_rules,
            tool_section=tool_section,
        )
        rule_images = rule_images_in_prompt_order(case_rules, gt_labels)
        result["rule_image_count"] = len(rule_images)
        all_images = list(audit_input.images or []) + tool_images + rule_images
        result["rendered_prompt_chars"] = len(rendered_prompt)
        result["rendered_prompt_sha256"] = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
        if args.save_prompt:
            result["rendered_prompt"] = rendered_prompt

        prepared = modules["prepare_accessible_multimodal_inputs"](
            rendered_prompt,
            all_images,
            image_max_tokens=args.image_max_tokens,
        )
        modules["assert_image_alignment"](
            prepared.text,
            prepared.images,
            f"rule consistency judge row {row_idx}",
        )
        result["dropped_images"] = prepared.dropped_images
        result["converted_images"] = prepared.converted_images
        result["final_image_count"] = len(prepared.images)
        if args.save_prompt:
            result["prepared_prompt"] = prepared.text

        user_message = modules["make_multimodal_message"](
            "user",
            prepared.text,
            prepared.images,
            image_max_tokens=args.image_max_tokens,
        )
        if args.image_transport == "base64" and isinstance(user_message.get("content"), list):
            local_urls = [
                part.get("image_url", {}).get("url", "")
                for part in user_message["content"]
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]
            if any(url.startswith("file://") for url in local_urls):
                raise ValueError(
                    "at least one image exceeded the base64 cap or could not be encoded; "
                    "increase --image-base64-max-bytes"
                )

        if args.dry_run:
            result.update(
                {
                    "success": True,
                    "dry_run": True,
                    "strict_parse_valid": None,
                    "latency_seconds": time.monotonic() - started,
                }
            )
            return result

        body: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system_prompt},
                user_message,
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if not args.omit_thinking_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": bool(args.thinking)}
        body.update(extra_body)
        api_response, retry_count = call_model(
            endpoint or "",
            headers,
            body,
            args.connect_timeout,
            args.timeout,
            args.max_retries,
        )
        result["http_success"] = True
        result["retry_count"] = retry_count
        choices = api_response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ValueError("API response contains no choices[0]")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = extract_message_text(message.get("content"))
        reasoning = extract_message_text(
            message.get("reasoning_content")
            if message.get("reasoning_content") is not None
            else message.get("reasoning")
        )
        parsed, parse_error = parse_output_json(content)
        validation_errors = (
            validate_judge_output(parsed, gt_labels, candidate_labels)
            if parsed is not None
            else []
        )
        output_warnings: list[str] = []
        if parsed and isinstance(parsed.get("brief_reason"), str) and len(parsed["brief_reason"]) > 200:
            output_warnings.append(
                f"brief_reason exceeds the requested 200 characters ({len(parsed['brief_reason'])})"
            )
        strict_error_parts = [value for value in [parse_error, *validation_errors] if value]
        strict_valid = not strict_error_parts
        result.update(
            {
                "raw_response": content,
                "reasoning_content": reasoning,
                "strict_parse_valid": strict_valid,
                "strict_parse_error": "; ".join(strict_error_parts) if strict_error_parts else None,
                "strict_validation_errors": validation_errors,
                "output_warnings": output_warnings,
                "success": strict_valid,
                "error_type": None if strict_valid else "invalid_model_output",
                "error": None if strict_valid else "; ".join(strict_error_parts),
                "verdict": parsed.get("verdict") if parsed else None,
                "issue_type": parsed.get("issue_type") if parsed else None,
                "key_rule_label": (
                    canonical_output_label(parsed.get("key_rule_label"), candidate_labels)
                    if parsed
                    else None
                ),
                "key_rule_label_raw": parsed.get("key_rule_label") if parsed else None,
                "brief_reason": parsed.get("brief_reason") if parsed else None,
                "usage": api_response.get("usage") if isinstance(api_response.get("usage"), dict) else {},
                "finish_reason": choice.get("finish_reason"),
                "response_id": api_response.get("id"),
                "response_model": api_response.get("model"),
            }
        )
    except Exception as exc:
        result["error_type"] = result.get("error_type") or "case_processing_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["latency_seconds"] = time.monotonic() - started
    return result


def percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summarize(results_path: Path, selected_count: int, args: argparse.Namespace) -> dict[str, Any]:
    latest = load_latest_results(results_path)
    rows = list(latest.values())
    verdicts = Counter(row.get("verdict") for row in rows if row.get("verdict"))
    issues = Counter(row.get("issue_type") for row in rows if row.get("issue_type"))
    errors = Counter(row.get("error_type") for row in rows if row.get("error_type"))
    per_source: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_gt_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "")].append(row)
        by_gt_type[str(row.get("gt_type") or "unknown")].append(row)

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total": len(group_rows),
            "success": sum(bool(row.get("success")) for row in group_rows),
            "strict_parse_valid": sum(bool(row.get("strict_parse_valid")) for row in group_rows),
            "verdicts": dict(Counter(row.get("verdict") for row in group_rows if row.get("verdict"))),
            "issue_types": dict(Counter(row.get("issue_type") for row in group_rows if row.get("issue_type"))),
        }

    for source, group_rows in sorted(grouped.items()):
        per_source[source] = group_summary(group_rows)
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    prompt_tokens = [
        int((row.get("usage") or {}).get("prompt_tokens"))
        for row in rows
        if (row.get("usage") or {}).get("prompt_tokens") is not None
    ]
    completion_tokens = [
        int((row.get("usage") or {}).get("completion_tokens"))
        for row in rows
        if (row.get("usage") or {}).get("completion_tokens") is not None
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input": str(args.input),
        "results_path": str(results_path),
        "selected_rows": selected_count,
        "unique_result_rows": len(rows),
        "success": sum(bool(row.get("success")) for row in rows),
        "failure": sum(not bool(row.get("success")) for row in rows),
        "http_success": sum(bool(row.get("http_success")) for row in rows),
        "dry_run_rows": sum(bool(row.get("dry_run")) for row in rows),
        "strict_parse_valid": sum(bool(row.get("strict_parse_valid")) for row in rows),
        "verdicts": dict(verdicts),
        "issue_types": dict(issues),
        "error_types": dict(errors),
        "by_gt_type": {key: group_summary(value) for key, value in sorted(by_gt_type.items())},
        "per_source": per_source,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "usage": {
            "prompt_tokens_total": sum(prompt_tokens),
            "completion_tokens_total": sum(completion_tokens),
        },
        "images": {
            "post_total": sum(int(row.get("post_image_count") or 0) for row in rows),
            "tool_total": sum(int(row.get("tool_image_count") or 0) for row in rows),
            "rule_total": sum(int(row.get("rule_image_count") or 0) for row in rows),
            "final_total": sum(int(row.get("final_image_count") or 0) for row in rows),
            "dropped_total": sum(len(row.get("dropped_images") or []) for row in rows),
            "converted_total": sum(len(row.get("converted_images") or []) for row in rows),
        },
        "preparation": {
            "rows_with_missing_tool_cache": sum(bool(row.get("tool_cache_missing")) for row in rows),
            "rows_with_tool_cache_errors": sum(bool(row.get("tool_cache_errors")) for row in rows),
            "rows_with_missing_rules": sum(bool(row.get("rule_missing")) for row in rows),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.prompt_template.is_file():
        raise FileNotFoundError(args.prompt_template)
    if not args.rule_cache.is_file():
        raise FileNotFoundError(args.rule_cache)
    if not args.tool_cache.is_dir():
        raise FileNotFoundError(args.tool_cache)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.image_max_tokens < 1:
        raise ValueError("--image-max-tokens must be >= 1")
    if not args.dry_run and (not args.base_url or not args.model):
        raise ValueError("--base-url and --model are required unless --dry-run is used")

    sys.path.insert(0, str(args.audit_project_root))
    from jinja2 import Environment, StrictUndefined
    from audit_agentic.agents.multimodal import (
        assert_image_alignment,
        make_multimodal_message,
        prepare_accessible_multimodal_inputs,
    )
    from audit_agentic.environment.tool_registry import DEFAULT_TOOL_REGISTRY
    from audit_agentic.schemas import AuditInput

    missing_registry_tools = [name for name in REQUIRED_TOOL_NAMES if name not in DEFAULT_TOOL_REGISTRY]
    if missing_registry_tools:
        raise ValueError(f"Required tools are absent from DEFAULT_TOOL_REGISTRY: {missing_registry_tools}")
    judge_registry = {name: DEFAULT_TOOL_REGISTRY[name] for name in REQUIRED_TOOL_NAMES}

    os.environ["AUDIT_SKIP_UNAVAILABLE_IMAGES"] = "1"
    os.environ["AUDIT_IMAGES_AS_BASE64"] = "1" if args.image_transport == "base64" else "0"
    os.environ["AUDIT_IMAGES_BASE64_MAX_BYTES"] = str(args.image_base64_max_bytes)

    template_text = args.prompt_template.read_text(encoding="utf-8")
    template = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    ).from_string(template_text)
    rules_by_source = load_rules(args.rule_cache)
    row_indices = parse_row_indices(args.row_indices)
    sources = set(args.sources) if args.sources else None
    selected_rows = load_selected_rows(args.input, sources, row_indices, args.limit)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    selected_rows = [
        item for item in selected_rows if item[0] % args.num_shards == args.shard_index
    ]
    if not selected_rows:
        raise ValueError("No input rows matched the selection")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPORT_ROOT / "outputs/rule_consistency_judge" / f"{args.input.stem}-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"

    endpoint = None if args.dry_run else normalize_endpoint(args.base_url)
    headers = parse_headers(args.header, args.api_key, args.api_key_header)
    extra_body = load_extra_body(args.extra_body_json)
    public_headers = {
        key: ("***" if key.lower() in {"authorization", "api-key", "x-api-key"} else value)
        for key, value in headers.items()
    }
    run_config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input": str(args.input),
        "prompt_template": str(args.prompt_template),
        "prompt_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
        "rule_cache": str(args.rule_cache),
        "tool_cache": str(args.tool_cache),
        "tool_cache_version": args.tool_cache_version,
        "tools": list(REQUIRED_TOOL_NAMES),
        "audit_project_root": str(args.audit_project_root),
        "output_dir": str(output_dir),
        "endpoint": endpoint,
        "model": args.model,
        "headers": public_headers,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "connect_timeout": args.connect_timeout,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "thinking": args.thinking,
        "omit_thinking_kwargs": args.omit_thinking_kwargs,
        "image_max_tokens": args.image_max_tokens,
        "image_transport": args.image_transport,
        "selected_rows": len(selected_rows),
        "sources": sorted(sources) if sources else None,
        "row_indices": args.row_indices,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume": args.resume,
        "retry_failed": args.retry_failed,
        "dry_run": args.dry_run,
        "save_prompt": args.save_prompt,
        "extra_body": extra_body,
    }
    atomic_write_json(run_config_path, run_config)

    if not args.resume and results_path.exists():
        results_path.write_text("", encoding="utf-8")
    existing = load_latest_results(results_path) if args.resume else {}
    pending = [
        (row_idx, row)
        for row_idx, row in selected_rows
        if not should_skip_existing(existing.get(row_idx), args.retry_failed)
    ]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected": len(selected_rows),
                "already_complete": len(selected_rows) - len(pending),
                "pending": len(pending),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    modules = {
        "AuditInput": AuditInput,
        "assert_image_alignment": assert_image_alignment,
        "make_multimodal_message": make_multimodal_message,
        "prepare_accessible_multimodal_inputs": prepare_accessible_multimodal_inputs,
    }
    completed = 0
    success = 0
    failed = 0
    if pending:
        with results_path.open("a", encoding="utf-8") as output_handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_to_idx = {
                    executor.submit(
                        process_case,
                        row_idx,
                        row,
                        args,
                        template,
                        rules_by_source,
                        judge_registry,
                        modules,
                        endpoint,
                        headers,
                        extra_body,
                    ): row_idx
                    for row_idx, row in pending
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    row_idx = future_to_idx[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "row_idx": row_idx,
                            "success": False,
                            "error_type": "unhandled_worker_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    completed += 1
                    if result.get("success"):
                        success += 1
                    else:
                        failed += 1
                    if completed % max(1, args.progress_every) == 0 or completed == len(pending):
                        print(
                            json.dumps(
                                {
                                    "completed": completed,
                                    "pending_total": len(pending),
                                    "success": success,
                                    "failed": failed,
                                    "last_row_idx": row_idx,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

    summary = summarize(results_path, len(selected_rows), args)
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
