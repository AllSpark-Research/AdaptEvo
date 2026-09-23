"""Verify the real AgentScope/OpenAI wire payload without model API access."""
from argparse import Namespace
import json
import unittest
from unittest.mock import patch

import httpx
import openai
from agentscope.message import TextBlock, UserMsg

from audit_agentic.eval.run_agentscope_agent_eval import build_model


class ReasoningEffortWireTests(unittest.IsolatedAsyncioTestCase):
    async def check_payload(self, effort):
        captured = []

        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "Kimi-K3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        args = Namespace(temperature=None, max_tokens=None, thinking=True,
                         parallel_tool_calls=False, timeout=2)
        payload = {"chat_template_kwargs": {"enable_thinking": True}}
        if effort is not None:
            payload["reasoning_effort"] = effort
        config = {"main_agent": {"api_base": "http://model.invalid/v1", "api_key": "test-key",
                                 "model": "Kimi-K3", "max_tokens": 8192, "extra_payload": payload}}
        real_client = openai.AsyncClient
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as http_client:
            def make_client(**kwargs):
                return real_client(**kwargs, http_client=http_client)

            with patch("openai.AsyncClient", side_effect=make_client):
                model, public = build_model(args, config)
            try:
                await model._call_api("Kimi-K3", [UserMsg(name="user", content=[TextBlock(text="test")])])
            finally:
                await model.client.close()
        self.assertEqual(len(captured), 1)
        body = captured[0]
        self.assertEqual(body["model"], "Kimi-K3")
        self.assertEqual(body["max_completion_tokens"], 8192)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("reasoning_effort", body["chat_template_kwargs"])
        if effort is None:
            self.assertNotIn("reasoning_effort", body)
            self.assertNotIn("reasoning_effort", public)
        else:
            self.assertEqual(body["reasoning_effort"], effort)
            self.assertEqual(public["reasoning_effort"], effort)

    async def test_default_does_not_send_effort(self):
        await self.check_payload(None)

    async def test_explicit_efforts_reach_wire_and_public_config(self):
        for effort in ("low", "medium", "high", "max", "xhigh", "vendor_custom"):
            with self.subTest(effort=effort):
                await self.check_payload(effort)


if __name__ == "__main__":
    unittest.main()
