"""BaseAgent: shared LLM/prompt/parse logic for all role agents."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from ..prompts.loader import PromptTemplateLoader
from ..schemas import AgentTrace
from .multimodal import build_multimodal_content, make_multimodal_message, DEFAULT_IMAGE_MAX_TOKENS


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    if not text:
        return text
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _content_to_text(content: Any) -> str:
    """Flatten a message ``content`` (str or OpenAI multimodal list) to plain
    text for trace/logging. Image parts become ``<image>`` placeholders.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "image_url":
                parts.append("<image>")
        return "".join(parts)
    return str(content) if content is not None else ""


def _extract_first_json_object(text: str) -> Optional[str]:
    """Best-effort: pull the first balanced {...} from text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    return str(getattr(response, "content", "") or "")


def _attach_llm_response_meta(trace: AgentTrace, response: Any) -> None:
    usage = getattr(response, "usage", None) or {}
    if not usage:
        return
    try:
        trace.model_extra["usage"] = usage  # type: ignore[attr-defined]
        trace.model_extra["prompt_tokens"] = usage.get("prompt_tokens")  # type: ignore[attr-defined]
        trace.model_extra["completion_tokens"] = usage.get("completion_tokens")  # type: ignore[attr-defined]
        trace.model_extra["total_tokens"] = usage.get("total_tokens")  # type: ignore[attr-defined]
        trace.model_extra["reasoning_tokens"] = usage.get("reasoning_tokens")  # type: ignore[attr-defined]
        trace.model_extra["finish_reason"] = getattr(response, "finish_reason", None)  # type: ignore[attr-defined]
        trace.model_extra["response_id"] = getattr(response, "response_id", None)  # type: ignore[attr-defined]
    except Exception:
        pass


class BaseAgent:
    """Render prompt -> call LLM -> parse JSON -> return dict + trace."""

    def __init__(
        self,
        llm_client: Any,
        prompt_loader: PromptTemplateLoader,
        name: str,
    ):
        self.llm = llm_client
        self.prompt_loader = prompt_loader
        self.name = name
        self.traces: list[AgentTrace] = []

    def build_prompt(self, template_name: str, **kwargs: Any) -> str:
        return self.prompt_loader.render(template_name, **kwargs)

    def parse_json_output(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        candidate = _strip_json_fence(text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # fallback: extract first balanced object
        obj_text = _extract_first_json_object(candidate)
        if obj_text is None:
            return {}
        try:
            return json.loads(obj_text)
        except json.JSONDecodeError:
            return {}

    def chat_with_meta(self, messages: list[Dict[str, Any]], **kwargs: Any) -> Any:
        if hasattr(self.llm, "chat_with_meta"):
            return self.llm.chat_with_meta(messages, **kwargs)
        return self.llm.chat(messages, **kwargs)

    def run_template(
        self,
        template_name: str,
        role_tag: str,
        images: list | None = None,
        image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
        response_format: Optional[Dict[str, Any]] = None,
        **render_kwargs: Any,
    ) -> tuple[Dict[str, Any], AgentTrace]:
        prompt = self.build_prompt(template_name, **render_kwargs)
        messages = [make_multimodal_message("user", prompt, images, image_max_tokens)]
        t0 = time.time()
        error: Optional[str] = None
        raw = ""
        parsed: Dict[str, Any] = {}
        response: Any = None
        try:
            call_kwargs: Dict[str, Any] = {"role_tag": role_tag}
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            response = self.chat_with_meta(messages, **call_kwargs)
            raw = _response_content(response)
            parsed = self.parse_json_output(raw)
            if not parsed:
                error = "json_parse_failed"
        except Exception as exc:  # noqa: BLE001 - surface any client error
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000.0
        trace = AgentTrace(
            role=self.name,
            template_name=template_name,
            rendered_prompt=prompt,
            raw_response=raw,
            parsed=parsed,
            error=error,
            latency_ms=latency_ms,
        )
        _attach_llm_response_meta(trace, response)
        self.traces.append(trace)
        return parsed, trace

    def run_messages(
        self,
        messages: list[Dict[str, Any]],
        role_tag: str,
        template_name: str = "<multi-turn>",
        parse_json: bool = True,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], AgentTrace]:
        """Multi-turn variant: send a pre-built messages list (e.g. continuation
        of a previous router turn). The final user message is treated as the
        "rendered_prompt" in the trace.
        """
        t0 = time.time()
        error: Optional[str] = None
        raw = ""
        parsed: Dict[str, Any] = {}
        response: Any = None
        try:
            call_kwargs: Dict[str, Any] = {"role_tag": role_tag}
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            response = self.chat_with_meta(messages, **call_kwargs)
            raw = _response_content(response)
            if parse_json:
                parsed = self.parse_json_output(raw)
                if not parsed:
                    error = "json_parse_failed"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000.0

        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = _content_to_text(m.get("content", ""))
                break

        trace = AgentTrace(
            role=self.name,
            template_name=template_name,
            rendered_prompt=last_user,
            raw_response=raw,
            parsed=parsed,
            error=error,
            latency_ms=latency_ms,
        )
        _attach_llm_response_meta(trace, response)
        # store full conversation for downstream tracing
        try:
            trace.model_extra["full_messages"] = messages  # type: ignore[attr-defined]
        except Exception:
            pass
        self.traces.append(trace)
        return parsed, trace
