"""Native Gemini wire regression tests, without production API access."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from agentscope.message import (AssistantMsg, Base64Source, DataBlock, Msg,
                               SystemMsg, TextBlock, ToolResultBlock, UserMsg)
from audit_agentic.eval.maas_gemini_model import MaasGeminiChatModel


def native(parts=None, reason="STOP"):
    return {"responseId": "response-test", "candidates": [{"finishReason": reason,
        "content": {"role": "model", "parts": parts or [{"text": "ok", "thoughtSignature": "text-signature"}]}}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3,
                          "thoughtsTokenCount": 5, "totalTokenCount": 15}}


def model(**kwargs):
    return MaasGeminiChatModel(endpoint="http://unused/native", api_key="fake-key",
                               model="Gemini 3.8 Flash", max_tokens=10000, **kwargs)


class GeminiWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_images_signature_and_args_roundtrip(self):
        m = model(effort="high")
        pic = DataBlock(source=Base64Source(media_type="image/png", data="cGlj"))
        user = UserMsg(name="u", content=[TextBlock(text="inspect"), pic])
        parts = [{"text": "reason", "thought": True},
                 {"functionCall": {"name": "inspect", "args": {"inline_data": {"mime_type": "unchanged"}}},
                  "thoughtSignature": "tool-signature"}]
        m._post = AsyncMock(return_value=native(parts))
        response = await m([user])
        self.assertEqual(response.usage.output_tokens, 8)
        saved = AssistantMsg(name="a", content=response.content)
        saved = Msg.model_validate_json(saved.model_dump_json())
        call = saved.content[-1]
        tool = AssistantMsg(name="a", content=[ToolResultBlock(id=call.id, name=call.name,
            output=[TextBlock(text="result"), pic])])
        payload = await m._build_payload(m.model, [SystemMsg(name="system", content="system"), user, saved, tool])
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "system"}]})
        self.assertEqual(payload["contents"][1]["parts"], parts)
        self.assertEqual(payload["contents"][0]["parts"][1]["inlineData"]["mimeType"], "image/png")
        result_parts = payload["contents"][2]["parts"]
        self.assertEqual(result_parts[0]["functionResponse"]["name"], "inspect")
        self.assertNotIn("id", result_parts[0]["functionResponse"])
        self.assertTrue(any("inlineData" in p for p in result_parts))
        self.assertEqual(payload["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH")
        self.assertNotIn("chat_template_kwargs", payload)

    async def test_text_signature_and_native_call_id_preserved(self):
        m = model()
        parts = [{"text": "hello", "thoughtSignature": "text-sig"},
                 {"functionCall": {"name": "f", "id": "native-1", "args": {}}, "thoughtSignature": "call-sig"}]
        m._post = AsyncMock(return_value=native(parts))
        response = await m([UserMsg(name="u", content="test")])
        payload = await m._build_payload(m.model, [AssistantMsg(name="a", content=response.content)])
        self.assertEqual(payload["contents"][0]["parts"], parts)

    async def test_structured_output(self):
        m = model()
        m._post = AsyncMock(return_value=native([{"functionCall": {"name": "generate_structured_output",
            "args": {"label": "pass"}}, "thoughtSignature": "signed"}]))
        schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
        result = await m.generate_structured_output([UserMsg(name="u", content="audit")], schema)
        self.assertEqual(result.content, {"label": "pass"})
        payload = m._post.call_args.args[0]
        self.assertEqual(payload["toolConfig"]["functionCallingConfig"],
                         {"mode": "ANY", "allowedFunctionNames": ["generate_structured_output"]})
        self.assertEqual(payload["tools"][0]["functionDeclarations"][0]["parameters"], schema)

    async def test_concurrent_history_isolation(self):
        m = model()
        async def echo(payload):
            await asyncio.sleep(0)
            return native([{"text": payload["contents"][0]["parts"][0]["text"]}])
        m._post = echo
        results = await asyncio.gather(*[m([UserMsg(name="u", content=str(i))]) for i in range(16)])
        self.assertEqual([r.content[0].text for r in results], [str(i) for i in range(16)])

    async def test_truncation_retry_and_safety_failure(self):
        m = model(max_retries=1, retry_delay=0)
        m._post = AsyncMock(side_effect=[native(reason="MAX_TOKENS"), native()])
        await m([UserMsg(name="u", content="test")])
        self.assertEqual(m._post.await_count, 2)
        m._post = AsyncMock(return_value=native(reason="SAFETY"))
        with self.assertRaisesRegex(ValueError, "SAFETY"):
            await m([UserMsg(name="u", content="test")])

    async def test_thinking_options(self):
        for enabled in (True, False):
            m = model(thinking=enabled)
            payload = await m._build_payload(m.model, [UserMsg(name="u", content="test")])
            config = payload["generationConfig"]["thinkingConfig"]
            self.assertEqual(config["includeThoughts"], enabled)
            self.assertNotIn("thinkingLevel", config)
            if enabled:
                self.assertNotIn("thinkingBudget", config)
            else:
                self.assertEqual(config["thinkingBudget"], 0)


if __name__ == "__main__":
    unittest.main()
