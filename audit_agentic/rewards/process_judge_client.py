"""Async multimodal client for the training-time Process Reward Judge."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import mimetypes
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from audit_agentic.agents.multimodal import (
    IMAGE_TAG,
    assert_image_alignment,
    prepare_accessible_multimodal_inputs,
)
from audit_agentic.rewards.process_judge import (
    PROCESS_JUDGE_RESPONSE_FORMAT,
    normalize_process_judge_output,
)


REPORT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_TEMPLATE = REPORT_ROOT / "prompt_templates/process_reward_judge.jinja2"
SYSTEM_PROMPT = (
    "你是一名内容审核 Agent 的 Process Reward Judge。严格依据用户消息中的完整审核轨迹"
    "进行五维评分。只做完成评分所需的必要核验，不要过度思考、重复推演或展开冗长分析；"
    "完成五维评分后立即停止思考，必须预留充足输出预算并只输出 JSON。"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def process_judge_enabled() -> bool:
    return _env_bool("AUDIT_PROCESS_JUDGE_ENABLED", False)


def _normalize_endpoint(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/,， ")
    if value.endswith("/chat/completions"):
        return value
    if not value:
        raise ValueError("AUDIT_PROCESS_JUDGE_BASE_URL is required")
    return f"{value}/chat/completions"


@dataclass(frozen=True)
class ProcessJudgeConfig:
    endpoint: str
    model: str
    prompt_template: Path
    api_key: str = ""
    api_key_header: str = "Authorization"
    thinking: bool = True
    reasoning_effort: str = ""
    max_tokens: int = 8192
    temperature: float = 0.0
    image_max_tokens: int = 448
    image_base64_max_bytes: int = 32 * 1024 * 1024
    connect_timeout: float = 10.0
    timeout: float = 600.0
    max_retries: int = 3
    log_path: str = ""

    @classmethod
    def from_env(cls) -> "ProcessJudgeConfig":
        return cls(
            endpoint=_normalize_endpoint(os.getenv("AUDIT_PROCESS_JUDGE_BASE_URL", "")),
            model=os.getenv("AUDIT_PROCESS_JUDGE_MODEL", "GLM-5.2-FP8"),
            prompt_template=Path(
                os.getenv(
                    "AUDIT_PROCESS_JUDGE_PROMPT_TEMPLATE",
                    str(DEFAULT_PROMPT_TEMPLATE),
                )
            ),
            api_key=os.getenv("AUDIT_PROCESS_JUDGE_API_KEY", ""),
            api_key_header=os.getenv(
                "AUDIT_PROCESS_JUDGE_API_KEY_HEADER",
                "Authorization",
            ),
            thinking=_env_bool("AUDIT_PROCESS_JUDGE_THINKING", True),
            reasoning_effort=os.getenv("AUDIT_PROCESS_JUDGE_REASONING_EFFORT", ""),
            max_tokens=_env_int("AUDIT_PROCESS_JUDGE_MAX_TOKENS", 8192),
            temperature=_env_float("AUDIT_PROCESS_JUDGE_TEMPERATURE", 0.0),
            image_max_tokens=_env_int("AUDIT_PROCESS_JUDGE_IMAGE_MAX_TOKENS", 448),
            image_base64_max_bytes=_env_int(
                "AUDIT_PROCESS_JUDGE_IMAGE_BASE64_MAX_BYTES",
                32 * 1024 * 1024,
            ),
            connect_timeout=_env_float("AUDIT_PROCESS_JUDGE_CONNECT_TIMEOUT", 10.0),
            timeout=_env_float("AUDIT_PROCESS_JUDGE_TIMEOUT", 600.0),
            max_retries=_env_int("AUDIT_PROCESS_JUDGE_MAX_RETRIES", 3),
            log_path=os.getenv("AUDIT_PROCESS_JUDGE_LOG_PATH", ""),
        )


def _source_to_path(source: Any) -> Optional[str]:
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


def _render_block(
    block: Any,
    images: list[str],
    image_origins: list[str],
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
        image = _source_to_path(block.get("source"))
        if image:
            images.append(image)
            image_origins.append(image_origin)
            return IMAGE_TAG
        return "[Unavailable Image]"
    if block_type == "hint":
        content = block.get("hint")
        if isinstance(content, list):
            return "\n".join(
                _render_block(item, images, image_origins, image_origin)
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
                _render_block(item, images, image_origins, "tool")
                for item in output
            )
        else:
            rendered_output = _render_block(
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
    compact = {
        key: value
        for key, value in block.items()
        if key not in {"id", "created_at", "finished_at", "metadata"}
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def render_agent_process(
    *,
    trace: dict[str, Any],
    prediction: list[str],
    binary_decision: str,
    audit_trace: str,
    used_rules: list[str],
    used_tools: list[str],
) -> tuple[str, list[str], list[str]]:
    images: list[str] = []
    image_origins: list[str] = []
    sections = [
        "## Audit Agent System Prompt\n" + str(trace.get("system_prompt") or ""),
    ]
    router = trace.get("router")
    if isinstance(router, dict) and isinstance(router.get("context"), list):
        sections.append("## Router Context")
        for turn_idx, message in enumerate(router["context"], start=1):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            blocks = content if isinstance(content, list) else [content]
            rendered = "\n".join(
                _render_block(block, images, image_origins)
                for block in blocks
            )
            sections.append(
                f"### Router Turn {turn_idx}: role={message.get('role') or 'unknown'}\n"
                f"{rendered}"
            )
    for turn_idx, message in enumerate(trace.get("context") or [], start=1):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        name = str(message.get("name") or "")
        header = f"## Turn {turn_idx}: role={role}" + (f", name={name}" if name else "")
        content = message.get("content")
        blocks = content if isinstance(content, list) else [content]
        rendered = "\n".join(
            _render_block(block, images, image_origins)
            for block in blocks
        )
        sections.append(f"{header}\n{rendered}")
        structured = message.get("structured_output")
        if structured:
            sections.append(
                f"### Turn {turn_idx} Structured Output\n"
                + json.dumps(structured, ensure_ascii=False, sort_keys=True)
            )
    if trace.get("structured_output"):
        sections.append(
            "## Normalized Final Structured Output\n"
            + json.dumps(trace["structured_output"], ensure_ascii=False, sort_keys=True)
        )
    sections.append(
        "## Normalized Evaluation Record\n"
        + json.dumps(
            {
                "predict_label": prediction,
                "binary_decision": binary_decision,
                "audit_trace": audit_trace,
                "used_rules": used_rules,
                "used_tools": used_tools,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if len(images) != len(image_origins):
        raise ValueError("Rendered image origins are not aligned with images")
    return "\n\n".join(sections), images, image_origins


def _kept_origins(
    images: list[str],
    origins: list[str],
    dropped_images: list[str],
) -> tuple[list[str], list[str]]:
    dropped_counts = Counter(dropped_images)
    kept: list[str] = []
    dropped: list[str] = []
    for image_ref, origin in zip(images, origins):
        if dropped_counts[image_ref] > 0:
            dropped_counts[image_ref] -= 1
            dropped.append(origin)
        else:
            kept.append(origin)
    return kept, dropped


def _image_data_url(image_ref: str, max_bytes: int) -> str:
    if image_ref.startswith("data:image/"):
        return image_ref
    path = image_ref[len("file://") :] if image_ref.startswith("file://") else image_ref
    stat = os.stat(path)
    if max_bytes > 0 and stat.st_size > max_bytes:
        raise ValueError(
            f"Prepared Judge image exceeds base64 limit: {stat.st_size} > {max_bytes}: {path}"
        )
    with open(path, "rb") as handle:
        raw = handle.read()
    mime = mimetypes.guess_type(path)[0]
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _multimodal_user_message(
    text: str,
    images: list[str],
    max_bytes: int,
) -> dict[str, Any]:
    assert_image_alignment(text, images, "Process Judge prompt")
    if not images:
        return {"role": "user", "content": text}
    parts = text.split(IMAGE_TAG)
    content: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if part and part.strip():
            content.append({"type": "text", "text": part})
        if index < len(images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(images[index], max_bytes),
                    },
                }
            )
    return {"role": "user", "content": content}


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )
    return "" if content is None else str(content)


def _runtime_meta(trace: dict[str, Any], image_meta: dict[str, Any]) -> dict[str, Any]:
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
    payload = {key: trace.get(key) for key in keys}
    payload.update(image_meta)
    return payload


def _append_log(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        # Judge records can be much larger than the filesystem's atomic append
        # size. Multiple rollout processes share this JSONL on CloudFS, so an
        # unlocked write can interleave records and corrupt later analysis.
        fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(line.encode("utf-8"))
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("Process Judge log write made no progress")
            remaining = remaining[written:]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


class ProcessJudgeClient:
    def __init__(self, config: ProcessJudgeConfig) -> None:
        self.config = config
        if not config.prompt_template.is_file():
            raise FileNotFoundError(config.prompt_template)
        from jinja2 import Environment, StrictUndefined

        self._template = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        ).from_string(config.prompt_template.read_text(encoding="utf-8"))

    @classmethod
    def from_env(cls) -> "ProcessJudgeClient":
        return cls(ProcessJudgeConfig.from_env())

    def _build_request(
        self,
        *,
        trace: dict[str, Any],
        prediction: list[str],
        binary_decision: str,
        audit_trace: str,
        used_rules: list[str],
        used_tools: list[str],
        candidate_labels: list[str],
        human_cot: str,
        source_id: str,
        note_id: str,
        protocol_issues: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        process_text, process_images, origins = render_agent_process(
            trace=trace,
            prediction=prediction,
            binary_decision=binary_decision,
            audit_trace=audit_trace,
            used_rules=used_rules,
            used_tools=used_tools,
        )
        assert_image_alignment(process_text, process_images, "raw Process Judge trajectory")
        prepared = prepare_accessible_multimodal_inputs(
            process_text,
            process_images,
            image_max_tokens=self.config.image_max_tokens,
            force_local_resize=True,
        )
        kept_origins, dropped_origins = _kept_origins(
            process_images,
            origins,
            prepared.dropped_images,
        )
        if len(kept_origins) != len(prepared.images):
            raise ValueError("Prepared Process Judge image origins are misaligned")
        image_meta = {
            "source_post_image_count": origins.count("post"),
            "source_tool_image_count": origins.count("tool"),
            "post_image_count": kept_origins.count("post"),
            "tool_image_count": kept_origins.count("tool"),
            "final_image_count": len(prepared.images),
            "dropped_post_image_count": dropped_origins.count("post"),
            "dropped_tool_image_count": dropped_origins.count("tool"),
            "dropped_images": prepared.dropped_images,
        }
        rendered_prompt = self._template.render(
            candidate_labels=candidate_labels,
            available_tools=list(trace.get("tool_schemas") or []),
            human_cot=human_cot,
            agent_process=prepared.text,
            agent_runtime_meta=json.dumps(
                _runtime_meta(trace, image_meta),
                ensure_ascii=False,
            ),
            protocol_issues=json.dumps(protocol_issues, ensure_ascii=False),
            source_id=source_id,
            note_id=note_id,
            row_idx=None,
        )
        assert_image_alignment(
            rendered_prompt,
            prepared.images,
            "rendered Process Judge prompt",
        )
        user_message = _multimodal_user_message(
            rendered_prompt,
            prepared.images,
            self.config.image_base64_max_bytes,
        )
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                user_message,
            ],
            "stream": False,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": PROCESS_JUDGE_RESPONSE_FORMAT,
            "chat_template_kwargs": {
                "enable_thinking": self.config.thinking,
            },
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        return body, image_meta

    async def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        started = time.monotonic()
        body, image_meta = await asyncio.to_thread(self._build_request, **kwargs)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            if self.config.api_key_header.lower() == "authorization":
                headers[self.config.api_key_header] = f"Bearer {self.config.api_key}"
            else:
                headers[self.config.api_key_header] = self.config.api_key

        import httpx

        retry_statuses = {408, 409, 429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None
        retry_count = 0
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.config.timeout,
                connect=self.config.connect_timeout,
            )
        ) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(
                        self.config.endpoint,
                        headers=headers,
                        json=body,
                    )
                    if response.status_code in retry_statuses and attempt < self.config.max_retries:
                        retry_count += 1
                        await asyncio.sleep(min(15.0, 2**attempt + random.random()))
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    choices = payload.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        raise ValueError("Process Judge response has no choices[0]")
                    finish_reason = str(choices[0].get("finish_reason") or "")
                    if finish_reason == "length":
                        raise ValueError(
                            "Process Judge response was truncated "
                            "(finish_reason=length)"
                        )
                    message = choices[0].get("message") or {}
                    raw_response = _extract_text(message.get("content"))
                    normalized = normalize_process_judge_output(
                        raw_response,
                        strict=True,
                    )
                    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                    result = {
                        "success": True,
                        "scores": {
                            key: normalized[key]
                            for key in PROCESS_JUDGE_RESPONSE_FORMAT["json_schema"]["schema"]["required"]
                        },
                        "process_reward": normalized["process_reward"],
                        "latency_seconds": time.monotonic() - started,
                        "retry_count": retry_count,
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
                        "error": "",
                        **image_meta,
                    }
                    _append_log(
                        self.config.log_path,
                        {
                            "timestamp": time.time(),
                            "note_id": kwargs.get("note_id"),
                            "source_id": kwargs.get("source_id"),
                            **result,
                        },
                    )
                    return result
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
                    retry_count += 1
                    await asyncio.sleep(min(15.0, 2**attempt + random.random()))

        assert last_error is not None
        result = {
            "success": False,
            "scores": {},
            "process_reward": None,
            "latency_seconds": time.monotonic() - started,
            "retry_count": retry_count,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "error": f"{type(last_error).__name__}: {last_error}",
            **image_meta,
        }
        _append_log(
            self.config.log_path,
            {
                "timestamp": time.time(),
                "note_id": kwargs.get("note_id"),
                "source_id": kwargs.get("source_id"),
                **result,
            },
        )
        return result
