#!/usr/bin/env python3
"""Stress-test the Process Reward Judge on duplicated full-trace smoke cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = REPORT_ROOT / "prompt_templates/process_reward_judge.jinja2"
DEFAULT_SMOKE_ROWS = []
SYSTEM_PROMPT = (
    "你是一名内容审核 Agent 的 Process Reward Judge。严格依据用户消息中的完整审核轨迹"
    "进行五维评分。内部分析保持简洁，避免重复推演，必须预留输出预算并只输出模板要求的 JSON 对象。"
)
DIMENSIONS = (
    "factual_grounding",
    "rule_fidelity",
    "evidence_coverage",
    "tool_use",
    "decision_consistency",
)
IMAGE_TAG = "<image>"
_THREAD_LOCAL = threading.local()
PROCESS_REWARD_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "process_reward_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                name: {"type": "number", "minimum": 0, "maximum": 1}
                for name in DIMENSIONS
            },
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate two full AgentScope smoke traces and stress-test a Process Judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--smoke-row", type=Path, action="append", dest="smoke_rows", required=True)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--copies-per-case", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="Optional OpenAI reasoning_effort value; omitted when empty.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send chat_template_kwargs.enable_thinking for SGLang models.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--image-max-tokens", type=int, default=448)
    parser.add_argument("--image-base64-max-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--api-key")
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--save-prompt", action="store_true")
    return parser.parse_args()


def normalize_endpoint(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/,， ")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_input_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if line.strip():
                rows[row_idx] = json.loads(line)
    return rows


def load_smoke_row(path: Path) -> dict[str, Any]:
    latest: Optional[dict[str, Any]] = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                latest = json.loads(line)
    if latest is None:
        raise ValueError(f"No JSON rows in {path}")
    trace = latest.get("agentscope_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("context"), list):
        raise ValueError(f"Smoke row has no full agentscope_trace.context: {path}")
    return latest


def source_to_path(source: Any) -> Optional[str]:
    if not isinstance(source, dict):
        return None
    source_type = str(source.get("type") or "")
    if source_type == "url":
        url = str(source.get("url") or "")
        return url[len("file://") :] if url.startswith("file://") else url or None
    if source_type == "base64":
        data = str(source.get("data") or "")
        media_type = str(source.get("media_type") or "image/jpeg")
        return f"data:{media_type};base64,{data}" if data else None
    return None


def render_block(
    block: Any,
    images: list[str],
    image_origins: Optional[list[str]] = None,
    image_origin: str = "post",
) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return str(block)
    block_type = str(block.get("type") or "")
    if block_type == "text":
        return str(block.get("text") or "")
    if block_type == "thinking":
        return "[Visible Reasoning]\n" + str(block.get("thinking") or "")
    if block_type == "data":
        image = source_to_path(block.get("source"))
        if image:
            images.append(image)
            if image_origins is not None:
                image_origins.append(image_origin)
            return IMAGE_TAG
        return "[Unavailable Image]"
    if block_type == "hint":
        content = block.get("hint")
        if isinstance(content, list):
            return "\n".join(
                render_block(item, images, image_origins, image_origin)
                for item in content
            )
    if block_type == "tool_call":
        tool_input = block.get("input")
        if isinstance(tool_input, (dict, list)):
            tool_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        return "\n".join(
            [
                "[Tool Call]",
                f"name: {block.get('name') or ''}",
                f"state: {block.get('state') or ''}",
                "input:",
                str(tool_input or ""),
            ]
        )
    if block_type == "tool_result":
        output = block.get("output")
        if isinstance(output, list):
            rendered_output = "\n".join(
                render_block(item, images, image_origins, "tool")
                for item in output
            )
        else:
            rendered_output = render_block(
                output,
                images,
                image_origins,
                "tool",
            )
        return "\n".join(
            [
                "[Tool Result]",
                f"name: {block.get('name') or ''}",
                f"state: {block.get('state') or ''}",
                "output:",
                rendered_output,
            ]
        )
    # Keep future AgentScope block types auditable without runtime timestamps.
    compact = {
        key: value
        for key, value in block.items()
        if key not in {"id", "created_at", "finished_at"}
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def render_agent_process(
    smoke_row: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    trace = smoke_row["agentscope_trace"]
    images: list[str] = []
    image_origins: list[str] = []
    sections = [
        "## Audit Agent System Prompt\n" + str(trace.get("system_prompt") or ""),
    ]
    for turn_idx, message in enumerate(trace.get("context") or [], start=1):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        name = str(message.get("name") or "")
        header = f"## Turn {turn_idx}: role={role}" + (f", name={name}" if name else "")
        content = message.get("content")
        if isinstance(content, list):
            rendered = "\n".join(
                render_block(block, images, image_origins)
                for block in content
            )
        else:
            rendered = render_block(content, images, image_origins)
        sections.append(f"{header}\n{rendered}")
        structured = message.get("structured_output")
        if structured:
            sections.append(
                f"### Turn {turn_idx} Structured Output\n"
                + json.dumps(structured, ensure_ascii=False, sort_keys=True),
            )
    if trace.get("structured_output"):
        sections.append(
            "## Normalized Final Structured Output\n"
            + json.dumps(trace["structured_output"], ensure_ascii=False, sort_keys=True),
        )
    sections.append(
        "## Normalized Evaluation Record\n"
        + json.dumps(
            {
                "predict_label": smoke_row.get("predict_label"),
                "binary_decision": smoke_row.get("binary_decision"),
                "audit_trace": smoke_row.get("audit_trace"),
                "used_rules": smoke_row.get("used_rules"),
                "used_tools": smoke_row.get("used_tools"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    if len(images) != len(image_origins):
        raise ValueError(
            f"Rendered image origin count {len(image_origins)} does not match "
            f"image count {len(images)}"
        )
    return "\n\n".join(sections), images, image_origins


def human_cot(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("human_cot") or "").strip()


def runtime_meta(smoke_row: dict[str, Any]) -> dict[str, Any]:
    trace = smoke_row.get("agentscope_trace") or {}
    keys = (
        "mode",
        "framework_version",
        "rule_loading_mode",
        "loaded_rule_labels",
        "final_label_detail_loaded",
        "context_compression_enabled",
        "context_compression_count",
        "finished_reason",
    )
    return {key: trace.get(key) for key in keys}


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


def parse_json_object(content: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    stripped = str(content or "").strip()
    if not stripped:
        return None, "assistant content is empty"
    try:
        payload = json.loads(stripped)
        return (payload, None) if isinstance(payload, dict) else (None, "top-level JSON is not an object")
    except json.JSONDecodeError:
        pass
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.I | re.S)
    candidates = balanced_json_objects(fenced.group(1) if fenced else stripped)
    parsed = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    if len(parsed) == 1:
        return parsed[0], None
    return None, "no unambiguous JSON object found" if not parsed else "multiple JSON objects found"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )
    return "" if content is None else str(content)


def get_session(concurrency: int) -> Any:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        import requests

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=concurrency,
            pool_maxsize=concurrency,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def call_model(
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    retry_statuses = {408, 409, 429, 500, 502, 503, 504}
    last_error: Optional[Exception] = None
    for attempt in range(args.max_retries + 1):
        try:
            response = get_session(args.concurrency).post(
                endpoint,
                headers=headers,
                json=body,
                timeout=(args.connect_timeout, args.timeout),
            )
            if response.status_code in retry_statuses and attempt < args.max_retries:
                time.sleep(min(15.0, 2**attempt + random.random()))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("API response is not a JSON object")
            return payload, attempt
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(min(15.0, 2**attempt + random.random()))
    assert last_error is not None
    raise last_error


def percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    smoke_paths = args.smoke_rows or DEFAULT_SMOKE_ROWS
    if args.copies_per_case < 1 or args.concurrency < 1:
        raise ValueError("--copies-per-case and --concurrency must be positive")
    for path in [args.input, args.prompt_template, *smoke_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(REPORT_ROOT))
    agentscope_root = REPORT_ROOT / "agentscope/src"
    if agentscope_root.is_dir():
        sys.path.insert(0, str(agentscope_root))

    from jinja2 import Environment, StrictUndefined
    from audit_agentic.agents.multimodal import (
        assert_image_alignment,
        make_multimodal_message,
        prepare_accessible_multimodal_inputs,
    )
    from audit_agentic.rewards.process_judge import (
        ProcessJudgeOutputError,
        normalize_process_judge_output,
    )

    os.environ["AUDIT_SKIP_UNAVAILABLE_IMAGES"] = "1"
    os.environ["AUDIT_IMAGES_AS_BASE64"] = "1"
    os.environ["AUDIT_IMAGES_BASE64_MAX_BYTES"] = str(args.image_base64_max_bytes)

    endpoint = normalize_endpoint(args.base_url)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    template_text = args.prompt_template.read_text(encoding="utf-8")
    template = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    ).from_string(template_text)
    input_rows = load_input_rows(args.input)
    smoke_rows = [load_smoke_row(path) for path in smoke_paths]

    prepared_cases: list[dict[str, Any]] = []
    for case_index, (smoke_path, smoke_row) in enumerate(zip(smoke_paths, smoke_rows)):
        row_idx = int(smoke_row["row_idx"])
        if row_idx not in input_rows:
            raise ValueError(f"row_idx {row_idx} is absent from {args.input}")
        data_row = input_rows[row_idx]
        trace = smoke_row["agentscope_trace"]
        process_text, process_images, process_image_origins = render_agent_process(
            smoke_row
        )
        rendered_prompt = template.render(
            candidate_labels=list(smoke_row.get("candidate_labels") or []),
            available_tools=list(trace.get("tool_schemas") or []),
            human_cot=human_cot(data_row),
            agent_process=process_text,
            agent_runtime_meta=json.dumps(runtime_meta(smoke_row), ensure_ascii=False),
            protocol_issues=json.dumps(smoke_row.get("errors") or [], ensure_ascii=False),
            source_id=smoke_row.get("source"),
            note_id=smoke_row.get("note_id"),
            row_idx=row_idx,
        )
        assert_image_alignment(
            rendered_prompt,
            process_images,
            f"unprepared process reward smoke case {case_index}",
        )
        prepared = prepare_accessible_multimodal_inputs(
            rendered_prompt,
            process_images,
            image_max_tokens=args.image_max_tokens,
        )
        assert_image_alignment(
            prepared.text,
            prepared.images,
            f"process reward smoke case {case_index}",
        )
        dropped_counts = Counter(prepared.dropped_images)
        kept_image_origins: list[str] = []
        dropped_image_origins: list[str] = []
        for image_ref, origin in zip(process_images, process_image_origins):
            if dropped_counts[image_ref] > 0:
                dropped_counts[image_ref] -= 1
                dropped_image_origins.append(origin)
            else:
                kept_image_origins.append(origin)
        if len(kept_image_origins) != len(prepared.images):
            raise ValueError(
                f"Prepared image origin count {len(kept_image_origins)} does "
                f"not match final image count {len(prepared.images)}"
            )
        user_message = make_multimodal_message(
            "user",
            prepared.text,
            prepared.images,
            image_max_tokens=args.image_max_tokens,
        )
        prepared_cases.append(
            {
                "case_index": case_index,
                "smoke_path": str(smoke_path),
                "row_idx": row_idx,
                "note_id": smoke_row.get("note_id"),
                "source": smoke_row.get("source"),
                "prompt_chars": len(prepared.text),
                "prompt_sha256": hashlib.sha256(prepared.text.encode()).hexdigest(),
                "image_count": len(prepared.images),
                "source_post_image_count": process_image_origins.count("post"),
                "source_tool_image_count": process_image_origins.count("tool"),
                "post_image_count": kept_image_origins.count("post"),
                "tool_image_count": kept_image_origins.count("tool"),
                "final_image_count": len(prepared.images),
                "final_placeholder_count": prepared.text.count(IMAGE_TAG),
                "dropped_post_image_count": dropped_image_origins.count("post"),
                "dropped_tool_image_count": dropped_image_origins.count("tool"),
                "dropped_images": prepared.dropped_images,
                "converted_images": prepared.converted_images,
                "user_message": user_message,
                "rendered_prompt": prepared.text if args.save_prompt else None,
            }
        )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPORT_ROOT / "outputs/process_reward_judge" / f"kimi-k3-smoke-stability-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    log_config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"

    tasks = [
        {
            "request_index": case_index * args.copies_per_case + copy_index,
            "copy_index": copy_index,
            "case": prepared_cases[case_index],
        }
        for case_index in range(len(prepared_cases))
        for copy_index in range(args.copies_per_case)
    ]
    atomic_write_json(
        log_config_path,
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "input": str(args.input),
            "smoke_rows": [str(path) for path in smoke_paths],
            "prompt_template": str(args.prompt_template),
            "prompt_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
            "endpoint": endpoint,
            "model": args.model,
            "copies_per_case": args.copies_per_case,
            "total_requests": len(tasks),
            "concurrency": args.concurrency,
            "reasoning_effort": args.reasoning_effort,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "image_max_tokens": args.image_max_tokens,
            "response_format": PROCESS_REWARD_RESPONSE_FORMAT,
            "prepared_cases": [
                {key: value for key, value in case.items() if key not in {"user_message", "rendered_prompt"}}
                for case in prepared_cases
            ],
        },
    )

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        case = task["case"]
        result = {
            "request_index": task["request_index"],
            "copy_index": task["copy_index"],
            "case_index": case["case_index"],
            "row_idx": case["row_idx"],
            "note_id": case["note_id"],
            "source": case["source"],
            "success": False,
            "http_success": False,
            "strict_output": False,
            "error_type": None,
            "error": None,
            "latency_seconds": None,
            "retry_count": 0,
            "scores": None,
            "process_reward": None,
            "raw_response": "",
            "reasoning_content": "",
            "usage": {},
        }
        try:
            body: dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    copy.deepcopy(case["user_message"]),
                ],
                "stream": False,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "response_format": PROCESS_REWARD_RESPONSE_FORMAT,
            }
            if args.reasoning_effort:
                body["reasoning_effort"] = args.reasoning_effort
            body["chat_template_kwargs"] = {
                "enable_thinking": bool(args.thinking),
            }
            response, retry_count = call_model(endpoint, headers, body, args)
            result["http_success"] = True
            result["retry_count"] = retry_count
            choices = response.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("API response has no choices[0]")
            choice = choices[0]
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = extract_text(message.get("content"))
            reasoning = extract_text(
                message.get("reasoning_content")
                if message.get("reasoning_content") is not None
                else message.get("reasoning")
            )
            result["raw_response"] = content
            result["reasoning_content"] = reasoning
            result["usage"] = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            result["finish_reason"] = choice.get("finish_reason")
            parsed, parse_error = parse_json_object(content)
            if parse_error:
                raise ProcessJudgeOutputError(parse_error)
            normalized = normalize_process_judge_output(parsed or {})
            output_keys = set(parsed or {})
            expected_keys = set(DIMENSIONS)
            extra_keys = sorted(output_keys - expected_keys)
            missing_keys = sorted(expected_keys - output_keys)
            result["extra_output_keys"] = extra_keys
            result["missing_output_keys"] = missing_keys
            result["strict_output"] = not extra_keys and not missing_keys
            result["scores"] = {name: normalized[name] for name in DIMENSIONS}
            result["process_reward"] = normalized["process_reward"]
            result["success"] = True
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
        result["latency_seconds"] = time.monotonic() - started
        return result

    run_started = time.monotonic()
    completed = 0
    success = 0
    failed = 0
    with results_path.open("w", encoding="utf-8") as output_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {executor.submit(worker, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "request_index": task["request_index"],
                        "success": False,
                        "error_type": "unhandled_worker_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed += 1
                success += bool(result.get("success"))
                failed += not bool(result.get("success"))
                if completed % max(1, args.progress_every) == 0 or completed == len(tasks):
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(tasks),
                                "success": success,
                                "failed": failed,
                                "elapsed_seconds": round(time.monotonic() - run_started, 1),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    scores_by_case: dict[str, Any] = {}
    for case_index, grouped_rows in sorted(
        defaultdict(list, {
            index: [row for row in rows if row.get("case_index") == index]
            for index in range(len(prepared_cases))
        }).items()
    ):
        dimension_summary: dict[str, Any] = {}
        for name in (*DIMENSIONS, "process_reward"):
            values = [
                float(row[name] if name == "process_reward" else (row.get("scores") or {}).get(name))
                for row in grouped_rows
                if row.get("success")
                and (row.get(name) if name == "process_reward" else (row.get("scores") or {}).get(name))
                is not None
            ]
            dimension_summary[name] = {
                "mean": statistics.mean(values) if values else None,
                "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        tuples = Counter(
            tuple(round(float((row.get("scores") or {}).get(name)), 4) for name in DIMENSIONS)
            for row in grouped_rows
            if row.get("success")
        )
        scores_by_case[str(case_index)] = {
            "row_idx": prepared_cases[case_index]["row_idx"],
            "note_id": prepared_cases[case_index]["note_id"],
            "source": prepared_cases[case_index]["source"],
            "requests": len(grouped_rows),
            "success": sum(bool(row.get("success")) for row in grouped_rows),
            "strict_output": sum(bool(row.get("strict_output")) for row in grouped_rows),
            "unique_score_vectors": len(tuples),
            "most_common_score_vectors": [
                {"scores": list(vector), "count": count}
                for vector, count in tuples.most_common(10)
            ],
            "scores": dimension_summary,
        }

    elapsed = time.monotonic() - run_started
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "total_requests": len(rows),
        "success": sum(bool(row.get("success")) for row in rows),
        "failure": sum(not bool(row.get("success")) for row in rows),
        "http_success": sum(bool(row.get("http_success")) for row in rows),
        "strict_output": sum(bool(row.get("strict_output")) for row in rows),
        "requests_with_retry": sum(int(row.get("retry_count") or 0) > 0 for row in rows),
        "retry_count_total": sum(int(row.get("retry_count") or 0) for row in rows),
        "error_types": dict(Counter(row.get("error_type") for row in rows if row.get("error_type"))),
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": len(rows) / elapsed if elapsed else None,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "usage": {
            "prompt_tokens_total": sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in rows),
            "completion_tokens_total": sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in rows),
            "reasoning_tokens_total": sum(int((row.get("usage") or {}).get("reasoning_tokens") or 0) for row in rows),
        },
        "per_case": scores_by_case,
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
