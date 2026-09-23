"""LLM client abstraction.

Two implementations:

* `OpenAIChatClient`: thin httpx wrapper around an OpenAI-compatible
  /v1/chat/completions endpoint. No SDK dep, easy to swap.
* `MockLLMClient`: returns canned responses keyed by a role tag. Used by the
  demo so the workflow can be exercised without an API.

Both expose ``chat(messages, **kwargs) -> str``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx


@dataclass
class LLMConfig:
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3
    structured_output: bool = True
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LLMConfig":
        return cls(
            api_base=cfg.get("api_base", ""),
            api_key=cfg.get("api_key", "") or "",
            model=cfg.get("model", ""),
            temperature=float(cfg.get("temperature", 0.3)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            timeout=float(cfg.get("timeout", 60)),
            max_retries=int(cfg.get("max_retries", 3)),
            structured_output=bool(cfg.get("structured_output", True)),
            extra_headers=cfg.get("extra_headers", {}) or {},
            extra_payload=cfg.get("extra_payload", {}) or {},
        )


@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    message: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_content: Optional[str] = None
    model: Optional[str] = None
    response_id: Optional[str] = None
    finish_reason: Optional[str] = None


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client.

    Retries on:
      - httpx.RequestError / TimeoutException (network/timeout)
      - HTTP 5xx / 429
      - Empty response body (data["choices"][0]["message"]["content"] is "" or None)
    Up to ``config.max_retries`` attempts with exponential backoff (capped 8s).
    Non-retryable: 4xx (other than 429), JSON decode error after retries → raises last.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.Client(timeout=config.timeout)

    def chat(self, messages: List[Dict[str, Any]], **overrides: Any) -> str:
        return self.chat_with_meta(messages, **overrides).content

    def chat_with_meta(self, messages: List[Dict[str, Any]], **overrides: Any) -> LLMResponse:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": overrides.get("model", self.config.model),
            "messages": messages,
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }
        for k, v in self.config.extra_payload.items():
            payload.setdefault(k, v)
        for key in ("response_format", "tools", "tool_choice", "parallel_tool_calls"):
            if key in overrides and overrides[key] is not None:
                payload[key] = overrides[key]
        if "extra_payload" in overrides and isinstance(overrides["extra_payload"], dict):
            payload.update(overrides["extra_payload"])

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)

        max_retries = max(1, int(self.config.max_retries))
        last_error: Optional[str] = None
        for attempt in range(max_retries):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
                status = resp.status_code
                # retryable HTTP statuses
                if status == 429 or 500 <= status < 600:
                    last_error = f"http_{status}"
                    time.sleep(min(2 ** attempt, 8))
                    continue
                # other 4xx: do NOT retry, raise immediately
                if status >= 400:
                    resp.raise_for_status()
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                raw_content = message.get("content")
                content = raw_content if isinstance(raw_content, str) else ""
                tool_calls = message.get("tool_calls") or []
                reasoning_content = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or None
                )
                if not content and not tool_calls:
                    # Native function calling commonly returns content=null.
                    # It is only empty when neither content nor tool calls exist.
                    last_error = "empty_content"
                    time.sleep(min(2 ** attempt, 8))
                    continue
                return LLMResponse(
                    content=content,
                    usage=data.get("usage") or {},
                    raw_response=data,
                    message=dict(message),
                    tool_calls=list(tool_calls),
                    reasoning_content=(
                        str(reasoning_content) if reasoning_content is not None else None
                    ),
                    model=data.get("model"),
                    response_id=data.get("id"),
                    finish_reason=choice.get("finish_reason"),
                )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, 8))
                continue
        # Exhausted retries
        raise RuntimeError(
            f"LLM chat failed after {max_retries} retries (last_error={last_error})"
        )


class MockLLMClient:
    """Returns canned strings based on a "role" hint passed in chat kwargs.

    The audit_agentic BaseAgent forwards ``role_tag`` so the mock can route.
    Use ``register(role_tag, fn_or_str)`` to set responses.
    """

    def __init__(self):
        self._responses: Dict[str, Callable[[List[Dict[str, Any]]], str] | str] = {}

    def register(self, role_tag: str, response):
        self._responses[role_tag] = response

    def chat(self, messages: List[Dict[str, Any]], role_tag: str = "", **_: Any) -> str:
        time.sleep(0.001)  # keep latency math non-zero
        r = self._responses.get(role_tag)
        if r is None:
            return "{}"
        if callable(r):
            return r(messages)
        return r

    def chat_with_meta(self, messages: List[Dict[str, Any]], role_tag: str = "", **kwargs: Any) -> LLMResponse:
        content = self.chat(messages, role_tag=role_tag, **kwargs)
        return LLMResponse(content=content, usage={})


def build_client(config: LLMConfig) -> OpenAIChatClient:
    return OpenAIChatClient(config)
