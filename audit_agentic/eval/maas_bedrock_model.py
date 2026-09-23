"""AgentScope Anthropic messages over the MAAS Bedrock streaming gateway."""

import asyncio
import copy
import os
from datetime import datetime
from types import SimpleNamespace

import aiohttp

from agentscope.credential import AnthropicCredential
from agentscope.formatter import AnthropicChatFormatter
from agentscope.model import AnthropicChatModel
from agentscope.model._base import ChatModelBase

from .maas_bedrock_stream import read_claude_response


class RetryableGatewayError(RuntimeError):
    """Transient HTTP or incomplete stream failure, with no request secrets."""


class MaasBedrockChatModel(AnthropicChatModel):
    """Stateless across conversations; signatures live in AgentScope messages."""

    def __init__(self, *, endpoint, api_key, model, max_tokens, thinking=True,
                 effort=None, temperature=0.7, timeout=600, parallel_tool_calls=False,
                 max_retries=3, retry_delay=1.0):
        if effort is not None and not thinking:
            raise ValueError("Claude reasoning effort requires thinking")
        # Reuse native formatting/parsing without creating an Anthropic SDK client.
        ChatModelBase.__init__(
            self, credential=AnthropicCredential(api_key=api_key), model=model,
            parameters=self.Parameters(max_tokens=max_tokens, thinking_enable=thinking),
            stream=False, max_retries=max_retries, retry_delay=retry_delay,
            context_size=200000,
        )
        self.formatter = AnthropicChatFormatter()
        self.endpoint = endpoint
        self.effort = effort
        self.temperature = temperature
        self.timeout = timeout
        self.parallel_tool_calls = parallel_tool_calls

    @classmethod
    def _get_retryable_exceptions(cls):
        return (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
                asyncio.TimeoutError, RetryableGatewayError)

    async def _build_payload(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        # The native formatter loads local/remote images synchronously. Offload it
        # so one image download cannot block all concurrent evaluation cases.
        formatted = await asyncio.to_thread(lambda: asyncio.run(self.formatter.format(messages)))
        payload = {
            "model": model_name, "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.parameters.max_tokens,
            "thinking": {"type": "adaptive" if self.parameters.thinking_enable else "disabled"},
        }
        if self.effort is not None:
            payload["output_config"] = {"effort": self.effort}
        if not self.parameters.thinking_enable:
            payload["temperature"] = self.temperature
        # Only forward known native generation overrides, never OpenAI fields.
        for key in ("max_tokens", "stop_sequences"):
            if key in kwargs:
                payload[key] = kwargs[key]
        system, conversation = [], []
        for message in formatted:
            if message["role"] == "system":
                system.extend(message["content"])
            elif conversation and conversation[-1]["role"] == message["role"]:
                conversation[-1]["content"].extend(message["content"])
            else:
                conversation.append(copy.deepcopy(message))
        if system:
            payload["system"] = system
        payload["messages"] = conversation
        native_tools, choice = self._format_tools(tools, tool_choice)
        if native_tools:
            payload["tools"] = native_tools
            choice = choice or {"type": "auto"}
            if self.parameters.thinking_enable and choice["type"] in {"any", "tool"}:
                # Adaptive thinking rejects forced tools. Restrict the tool list
                # when a particular tool was requested; AgentScope adds guidance.
                if choice["type"] == "tool":
                    payload["tools"] = [t for t in native_tools if t["name"] == choice["name"]]
                choice = {"type": "auto"}
            if choice["type"] != "none":
                choice["disable_parallel_tool_use"] = not self.parallel_tool_calls
            payload["tool_choice"] = choice
        return payload

    async def _post(self, payload):
        timeout = aiohttp.ClientTimeout(total=self.timeout, sock_connect=min(30, self.timeout),
                                        sock_read=self.timeout)
        # Never attach the API key to image downloads or follow model redirects.
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.post(self.endpoint, json=payload, allow_redirects=False,
                                   headers={"api-key": self.credential.api_key.get_secret_value(),
                                            "Content-Type": "application/json"}) as response:
                if response.status != 200:
                    error = f"MAAS Bedrock HTTP {response.status}"
                    # Preserve actionable gateway errors, not headers or raw bodies.
                    try:
                        details = await response.json(content_type=None)
                        if isinstance(details, dict):
                            summary = ": ".join(str(details[k]) for k in ("error_type", "detail") if details.get(k))
                            summary = summary.replace(self.credential.api_key.get_secret_value(), "[REDACTED]")
                            if summary:
                                error += " - " + summary[:500]
                    except (ValueError, aiohttp.ClientError):
                        pass
                    if response.status in {408, 429} or response.status >= 500:
                        raise RetryableGatewayError(error)
                    raise ValueError(error)
                try:
                    return await read_claude_response(response)
                except (ValueError, RuntimeError) as exc:
                    raise RetryableGatewayError("Invalid or incomplete MAAS Bedrock stream") from exc

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        started = datetime.now()
        payload = await self._build_payload(model_name, messages, tools, tool_choice, **kwargs)
        native = await self._post(payload)
        if native.get("stop_reason") == "max_tokens":
            raise RetryableGatewayError("Claude output truncated at max_tokens")
        if native.get("stop_reason") not in {"end_turn", "tool_use", "stop_sequence"}:
            raise ValueError("Claude returned a non-success stop reason: " + str(native.get("stop_reason")))
        response = SimpleNamespace(
            id=native.get("id"),
            content=[SimpleNamespace(**block) for block in native["content"]],
            usage=SimpleNamespace(**native["usage"]) if native.get("usage") else None,
        )
        return await self._parse_anthropic_completion_response(started, response)


def build_maas_model(args, cfg):
    key_env = cfg.get("api_key_env") or "MAAS_API_KEY"
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"Set the credential environment variable {key_env}")
    temperature = cfg.get("temperature", 0.7) if args.temperature is None else args.temperature
    max_tokens = cfg.get("max_tokens", 10000) if args.max_tokens is None else args.max_tokens
    effort = (cfg.get("extra_payload", {}).get("output_config") or {}).get("effort")
    model = MaasBedrockChatModel(
        endpoint=cfg["api_base"], api_key=api_key, model=cfg["model"],
        max_tokens=max_tokens, thinking=bool(args.thinking), effort=effort,
        temperature=temperature, timeout=args.timeout,
        parallel_tool_calls=bool(args.parallel_tool_calls),
        max_retries=int(cfg.get("max_retries", 3)), retry_delay=float(cfg.get("retry_delay", 1.0)),
    )
    public = {
        "api_format": "maas-bedrock", "api_base": cfg["api_base"], "model": cfg["model"],
        "api_key_env": key_env, "temperature": None if args.thinking else temperature,
        "max_tokens": max_tokens, "thinking": bool(args.thinking),
        "reasoning_effort": effort, "thinking_type": "adaptive" if args.thinking else "disabled",
        "parallel_tool_calls": bool(args.parallel_tool_calls), "timeout": args.timeout,
    }
    return model, public
