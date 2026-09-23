"""Offline native-wire tests; no production model or media service is called."""

import asyncio
import base64
import json
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web
from agentscope.message import (AssistantMsg, Base64Source, DataBlock, TextBlock,
                               ToolResultBlock, UserMsg)
from agentscope.tool import ToolChoice

from audit_agentic.eval.maas_bedrock_model import MaasBedrockChatModel, RetryableGatewayError
from audit_agentic.eval.maas_bedrock_stream import read_claude_response
from audit_agentic.eval.run_agentscope_agent_eval import build_model


def native(content=None, stop="end_turn"):
    return {"type": "message", "id": "msg-test", "role": "assistant", "model": "claude opus 5",
            "content": content or [{"type": "text", "text": "ok"}], "stop_reason": stop,
            "usage": {"input_tokens": 7, "output_tokens": 4, "cache_read_input_tokens": 2}}


def model(**kwargs):
    return MaasBedrockChatModel(endpoint="http://unused.invalid/native", api_key="test-secret",
                                model="claude opus 5", max_tokens=10000, effort="high", **kwargs)


class NativeWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_thinking_tool_images_and_signature_roundtrip(self):
        m = model()
        picture = DataBlock(source=Base64Source(media_type="image/png", data="cGljdHVyZQ=="))
        user = UserMsg(name="user", content=[TextBlock(text="inspect"), picture])
        content = [{"type": "thinking", "thinking": "check", "signature": "signed-original"},
                   {"type": "redacted_thinking", "data": "opaque-original"},
                   {"type": "tool_use", "id": "call-1", "name": "check", "input": {"label": "x"}}]
        m._post = AsyncMock(return_value=native(content, "tool_use"))
        response = await m([user])
        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 4)
        self.assertEqual(response.content[0].signature, "signed-original")
        result = AssistantMsg(name="tool", content=[ToolResultBlock(
            id="call-1", name="check", output=[TextBlock(text="result"), picture])])
        payload = await m._build_payload(m.model, [user, AssistantMsg(name="a", content=response.content), result])
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertEqual(payload["max_tokens"], 10000)
        for key in ("temperature", "reasoning_effort", "chat_template_kwargs", "stream"):
            self.assertNotIn(key, payload)
        self.assertEqual(payload["messages"][1]["content"], content)
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image")
        self.assertEqual(payload["messages"][2]["content"][0]["content"][1]["type"], "image")

    async def test_structured_output_uses_native_auto_tool(self):
        m = model()
        m._post = AsyncMock(return_value=native([{"type": "tool_use", "id": "final-1",
            "name": "generate_structured_output", "input": {"label": "pass"}}], "tool_use"))
        schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
        response = await m.generate_structured_output([UserMsg(name="u", content="audit")], schema)
        self.assertEqual(response.content, {"label": "pass"})
        payload = m._post.call_args.args[0]
        self.assertEqual(payload["tool_choice"]["type"], "auto")
        self.assertEqual(payload["tools"][0]["input_schema"], schema)

    async def test_concurrent_requests_do_not_share_history(self):
        m = model()
        async def echo(payload):
            await asyncio.sleep(0)
            return native([{"type": "text", "text": payload["messages"][0]["content"][0]["text"]}])
        m._post = echo
        results = await asyncio.gather(*[m([UserMsg(name="u", content=str(i))]) for i in range(12)])
        self.assertEqual([r.content[0].text for r in results], [str(i) for i in range(12)])

    async def test_truncated_output_retries(self):
        m = model(max_retries=1, retry_delay=0)
        m._post = AsyncMock(side_effect=[native(stop="max_tokens"), native()])
        response = await m([UserMsg(name="u", content="test")])
        self.assertEqual(response.content[0].text, "ok")
        self.assertEqual(m._post.await_count, 2)

    async def test_disabled_thinking_and_forced_tool(self):
        m = MaasBedrockChatModel(endpoint="http://unused", api_key="test", model="claude opus 5",
                                 max_tokens=10000, thinking=False)
        payload = await m._build_payload(m.model, [UserMsg(name="u", content="test")],
            [{"type": "function", "function": {"name": "final", "parameters": {"type": "object"}}}],
            ToolChoice(mode="final"))
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["tool_choice"]["type"], "tool")
        self.assertEqual(payload["temperature"], 0.7)

    async def test_real_http_transport_and_redirect_rejection(self):
        seen = []
        async def handler(request):
            seen.append((request.headers.get("api-key"), await request.json()))
            return web.json_response(native())
        async def redirect(request):
            raise web.HTTPFound("/native")
        async def auth_error(request):
            return web.json_response({"error_type": "provider.auth_error",
                                      "detail": "Denied test-secret"}, status=400)
        app = web.Application()
        app.router.add_post("/native", handler)
        app.router.add_post("/redirect", redirect)
        app.router.add_post("/auth-error", auth_error)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            m = model()
            m.endpoint = f"http://127.0.0.1:{port}/native"
            await m([UserMsg(name="u", content="hi")])
            self.assertEqual(seen[0][0], "test-secret")
            self.assertEqual(seen[0][1]["anthropic_version"], "bedrock-2023-05-31")
            m.endpoint = f"http://127.0.0.1:{port}/redirect"
            with self.assertRaisesRegex(ValueError, "HTTP 302"):
                await m([UserMsg(name="u", content="hi")])
            self.assertEqual(len(seen), 1)
            m.endpoint = f"http://127.0.0.1:{port}/auth-error"
            with self.assertRaisesRegex(ValueError, "provider.auth_error") as error:
                await m([UserMsg(name="u", content="hi")])
            self.assertNotIn("test-secret", str(error.exception))
        finally:
            await runner.cleanup()

    async def test_fragmented_gateway_events(self):
        events = [
            {"type": "message_start", "message": native(content=[])},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "\u901a\u8fc7"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ]
        raw = b"".join(json.dumps({"chunk": {"bytes": base64.b64encode(json.dumps(e).encode()).decode()}}).encode() for e in events)
        async def chunks(data):
            for i in range(0, len(data), 7):
                yield data[i:i+7]
        result = await read_claude_response(SimpleNamespace(content=SimpleNamespace(iter_any=lambda: chunks(raw))))
        self.assertEqual(result["content"][0]["text"], "\u901a\u8fc7")
        self.assertEqual(result["usage"]["output_tokens"], 9)
        with self.assertRaises(ValueError):
            await read_claude_response(SimpleNamespace(content=SimpleNamespace(iter_any=lambda: chunks(raw[:-20]))))

    async def test_build_model_uses_env_without_exposing_secret(self):
        args = Namespace(temperature=None, max_tokens=None, thinking=True, parallel_tool_calls=False, timeout=10)
        config = {"main_agent": {"api_format": "maas-bedrock", "api_base": "http://unused/native",
            "model": "claude opus 5", "api_key_env": "TEST_MAAS_KEY", "max_tokens": 10000,
            "extra_payload": {"output_config": {"effort": "high"}}}}
        with patch.dict("os.environ", {"TEST_MAAS_KEY": "secret-not-to-save"}):
            m, public = build_model(args, config)
        self.assertIsInstance(m, MaasBedrockChatModel)
        self.assertNotIn("secret-not-to-save", json.dumps(public))
        self.assertEqual(public["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
