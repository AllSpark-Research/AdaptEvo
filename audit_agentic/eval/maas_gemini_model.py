"""AgentScope Gemini messages over the MAAS native generateContent endpoint."""

import asyncio
import base64
import json
import os
import uuid
from datetime import datetime

import aiohttp

from agentscope.credential import GeminiCredential
from agentscope.formatter import GeminiChatFormatter
from agentscope.message import TextBlock, ThinkingBlock, ToolCallBlock
from agentscope.model import GeminiChatModel
from agentscope.model._base import ChatModelBase
from agentscope.model._model_response import ChatResponse
from agentscope.model._model_usage import ChatUsage

from .maas_bedrock_model import RetryableGatewayError


def signed_id(signature=None, native_id=None):
    # Keep provider metadata in the block identity, which AgentScope preserves
    # through its event/state pipeline. No shared conversation cache is needed.
    value = {"signature": signature, "native_id": native_id, "nonce": uuid.uuid4().hex}
    return "maasgemini:" + base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def unpack_id(value):
    if not value.startswith("maasgemini:"):
        return None
    return json.loads(base64.urlsafe_b64decode(value.split(":", 1)[1]))


def rest_part(part):
    # Only rename protocol fields. Tool arguments and response objects are opaque.
    result = dict(part)
    if "inline_data" in result:
        data = result.pop("inline_data")
        result["inlineData"] = {"mimeType": data["mime_type"], "data": data["data"]}
    for before, after in (("function_call", "functionCall"), ("function_response", "functionResponse")):
        if before in result:
            result[after] = dict(result.pop(before))
            metadata = unpack_id(result[after].get("id", ""))
            if metadata:
                if metadata["native_id"]:
                    result[after]["id"] = metadata["native_id"]
                else:
                    result[after].pop("id", None)
                if after == "functionCall" and metadata["signature"]:
                    result["thoughtSignature"] = metadata["signature"]
    return result


class MaasGeminiChatModel(GeminiChatModel):
    def __init__(self, *, endpoint, api_key, model, max_tokens=10000,
                 thinking=True, effort=None, temperature=1.0, timeout=600,
                 max_retries=3, retry_delay=1.0):
        if not thinking and effort is not None:
            raise ValueError("Gemini reasoning effort requires thinking")
        ChatModelBase.__init__(
            self, credential=GeminiCredential(api_key=api_key), model=model,
            parameters=self.Parameters(max_tokens=max_tokens, thinking_enable=thinking,
                                       temperature=temperature),
            stream=False, max_retries=max_retries, retry_delay=retry_delay,
            context_size=1048576,
        )
        self.formatter = GeminiChatFormatter()
        self.endpoint = endpoint
        self.effort = effort
        self.timeout = timeout

    @classmethod
    def _get_retryable_exceptions(cls):
        return (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
                asyncio.TimeoutError, RetryableGatewayError)

    async def _format_native(self, messages):
        system, contents = [], []
        for msg in messages:
            for block in msg.get_content_blocks():
                formatted = await self.formatter.format([msg.model_copy(update={"content": [block]})])
                metadata = unpack_id(block.id)
                if not formatted and metadata and metadata["signature"]:
                    formatted = [{"role": "model", "parts": [{"thoughtSignature": metadata["signature"]}]}]
                for content in formatted:
                    parts = [rest_part(p) for p in content["parts"]]
                    if isinstance(block, (TextBlock, ThinkingBlock)) and metadata and metadata["signature"]:
                        parts[0]["thoughtSignature"] = metadata["signature"]
                    if msg.role == "system":
                        system.extend(parts)
                    elif contents and contents[-1]["role"] == content["role"]:
                        contents[-1]["parts"].extend(parts)
                    else:
                        contents.append({"role": content["role"], "parts": parts})
        return system, contents

    async def _build_payload(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        system, contents = await asyncio.to_thread(lambda: asyncio.run(self._format_native(messages)))
        thinking = {"includeThoughts": bool(self.parameters.thinking_enable)}
        if not self.parameters.thinking_enable:
            thinking["thinkingBudget"] = 0
        elif self.effort:
            thinking["thinkingLevel"] = self.effort.upper()
        generation = {"maxOutputTokens": self.parameters.max_tokens,
                      "temperature": self.parameters.temperature, "thinkingConfig": thinking}
        payload = {"model": model_name, "contents": contents, "generationConfig": generation}
        if system:
            payload["systemInstruction"] = {"parts": system}
        declarations, choice = self._format_tools(tools, tool_choice)
        if declarations:
            payload["tools"] = [{"functionDeclarations": t["function_declarations"]} for t in declarations]
        if choice:
            config = dict(choice["function_calling_config"])
            if "allowed_function_names" in config:
                config["allowedFunctionNames"] = config.pop("allowed_function_names")
            payload["toolConfig"] = {"functionCallingConfig": config}
        return payload

    async def _post(self, payload):
        timeout = aiohttp.ClientTimeout(total=self.timeout, sock_connect=min(30, self.timeout), sock_read=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.post(self.endpoint, json=payload, allow_redirects=False,
                                   headers={"api-key": self.credential.api_key.get_secret_value(),
                                            "Content-Type": "application/json"}) as response:
                try:
                    body = await response.json(content_type=None)
                except ValueError as exc:
                    raise RetryableGatewayError(f"MAAS Gemini HTTP {response.status}: invalid JSON response") from exc
                if response.status != 200 or (isinstance(body, dict) and body.get("error")):
                    details = body.get("error", body) if isinstance(body, dict) else {}
                    summary = ": ".join(str(details[k]) for k in ("error_type", "status", "message", "detail") if isinstance(details, dict) and details.get(k))
                    summary = summary.replace(self.credential.api_key.get_secret_value(), "[REDACTED]")[:500]
                    error = f"MAAS Gemini HTTP {response.status}: {summary}"
                    if response.status in {408, 429} or response.status >= 500:
                        raise RetryableGatewayError(error)
                    raise ValueError(error)
                return body

    def _parse_native(self, native, started):
        candidates = native.get("candidates") or []
        if len(candidates) != 1:
            raise ValueError("Gemini response must contain one candidate; may be blocked")
        candidate = candidates[0]
        reason = candidate.get("finishReason")
        if reason == "MAX_TOKENS":
            raise RetryableGatewayError("Gemini output truncated at maxOutputTokens")
        if reason != "STOP":
            raise ValueError("Gemini non-success finishReason: " + str(reason))
        blocks = []
        for part in (candidate.get("content") or {}).get("parts", []):
            signature = part.get("thoughtSignature")
            if "text" in part:
                attrs = {"id": signed_id(signature)} if signature else {}
                if part.get("thought"):
                    blocks.append(ThinkingBlock(thinking=part["text"], **attrs))
                else:
                    blocks.append(TextBlock(text=part["text"], **attrs))
            elif "functionCall" in part:
                call = part["functionCall"]
                blocks.append(ToolCallBlock(id=signed_id(signature, call.get("id")), name=call["name"],
                                            input=json.dumps(call.get("args") or {}, ensure_ascii=False)))
            elif signature:
                blocks.append(ThinkingBlock(thinking="", id=signed_id(signature)))
            else:
                raise ValueError("Unsupported Gemini response part")
        if not any(isinstance(b, (TextBlock, ToolCallBlock)) for b in blocks):
            raise ValueError("Gemini returned no answer or tool call")
        data = native.get("usageMetadata") or {}
        usage = None
        if data:
            output = data.get("candidatesTokenCount", 0) + data.get("thoughtsTokenCount", 0)
            usage = ChatUsage(input_tokens=data.get("promptTokenCount", 0), output_tokens=output,
                              cache_input_tokens=data.get("cachedContentTokenCount", 0),
                              time=(datetime.now() - started).total_seconds())
        return ChatResponse(id=native.get("responseId") or uuid.uuid4().hex, content=blocks,
                            usage=usage, is_last=True)

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        started = datetime.now()
        payload = await self._build_payload(model_name, messages, tools, tool_choice, **kwargs)
        return self._parse_native(await self._post(payload), started)


def build_maas_gemini_model(args, cfg):
    key_env = cfg.get("api_key_env") or "MAAS_API_KEY"
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f"Set the credential environment variable {key_env}")
    temperature = cfg.get("temperature", 1.0) if args.temperature is None else args.temperature
    max_tokens = cfg.get("max_tokens", 10000) if args.max_tokens is None else args.max_tokens
    effort = (cfg.get("extra_payload", {}).get("thinkingConfig") or {}).get("thinkingLevel")
    model = MaasGeminiChatModel(endpoint=cfg["api_base"], api_key=key, model=cfg["model"],
        max_tokens=max_tokens, thinking=bool(args.thinking), effort=effort,
        temperature=temperature, timeout=args.timeout, max_retries=int(cfg.get("max_retries", 3)),
        retry_delay=float(cfg.get("retry_delay", 1.0)))
    return model, {"api_format": "maas-gemini", "api_base": cfg["api_base"], "model": cfg["model"],
        "api_key_env": key_env, "temperature": temperature, "max_tokens": max_tokens,
        "thinking": bool(args.thinking), "reasoning_effort": effort, "timeout": args.timeout}
