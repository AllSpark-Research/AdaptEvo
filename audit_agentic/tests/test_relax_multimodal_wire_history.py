"""Stable multimodal tool references must keep Relax trajectories linear."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import (
    AssistantMsg, Base64Source, DataBlock, Msg, TextBlock,
    ToolCallBlock, ToolResultBlock, ToolResultState, URLSource, UserMsg,
)
from audit_agentic.relax_app.relax_wire_history import RelaxWireHistory


def _messages(source_kind="url"):
    def source(index):
        if source_kind == "base64":
            return Base64Source(data="aW1hZ2U=", media_type="image/png")
        return URLSource(url=f"https://example.com/{index}.png", media_type="image/png")

    return [
        UserMsg(name="user", content="Audit this post."),
        AssistantMsg(name="audit", content=[
            TextBlock(text="Check external evidence."),
            ToolCallBlock(id="call_1", name="get_recent_notes", input="{}"),
            ToolResultBlock(
                id="call_1", name="get_recent_notes", state=ToolResultState.SUCCESS,
                output=[
                    TextBlock(text="Two separate evidence images:"),
                    DataBlock(id="evidence_1", source=source(1)),
                    DataBlock(id="evidence_2", source=source(2)),
                ],
            ),
        ]),
    ]


@pytest.mark.parametrize("source_kind", ["url", "base64"])
@pytest.mark.parametrize("rebuild", ["same", "deepcopy", "json"])
def test_multimodal_tool_history_is_stable(source_kind, rebuild):
    async def run():
        formatter = OpenAIChatFormatter()
        messages = _messages(source_kind)
        original = await formatter.format(messages)
        if rebuild == "deepcopy":
            messages = copy.deepcopy(messages)
        elif rebuild == "json":
            messages = [Msg.model_validate_json(msg.model_dump_json()) for msg in messages]
        rebuilt = await formatter.format(messages)
        assert rebuilt == original
        assert "[evidence_1]" in rebuilt[2]["content"]
        assert "[evidence_2]" in rebuilt[2]["content"]
        image_parts = [p for p in rebuilt[3]["content"] if p["type"] == "image_url"]
        assert len(image_parts) == 2
        assert rebuilt[3]["content"][1]["text"].startswith("- evidence_1 ")
        assert rebuilt[3]["content"][3]["text"].startswith("- evidence_2 ")
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["tool_text", "image", "image_order"])
def test_changed_evidence_is_not_silently_canonicalized(mutation):
    async def run():
        formatter = OpenAIChatFormatter()
        messages = _messages()
        sent = await formatter.format(messages)
        response = {"choices": [{"message": {"role": "assistant", "content": "Review evidence."}}]}
        wire = RelaxWireHistory()
        wire.record(sent, response)
        changed = copy.deepcopy(messages)
        output = changed[1].get_content_blocks("tool_result")[0].output
        if mutation == "tool_text":
            output[0].text = "Materially different evidence."
        elif mutation == "image":
            output[1].source = URLSource(url="https://example.com/changed.png", media_type="image/png")
        else:
            output[1], output[2] = output[2], output[1]
        rebuilt = await formatter.format(changed)
        rebuilt.append(response["choices"][0]["message"])
        assert wire.reconcile(rebuilt) == rebuilt
        assert wire.rejected_turns == 1
        assert wire.canonicalized_turns == 0
    asyncio.run(run())


def test_multimodal_history_exports_all_generated_tokens_and_rejects_real_branch():
    from relax.agentic.session.service import AgenticSessionShard
    from relax.agentic.session.state import SessionForest, check_messages

    async def run():
        shard = AgenticSessionShard.__ray_metadata__.modified_class
        forest = SessionForest.create_empty(session_id="multimodal_regression")
        wire = RelaxWireHistory()
        formatter = OpenAIChatFormatter()
        source = _messages()
        messages = [source[0]]
        last_hash = None
        for turn in range(3):
            rebuilt = await formatter.format(messages)
            sent = wire.reconcile(rebuilt)
            normalized = check_messages(sent)
            parent, delta = shard._match_parent_state_hash(
                None, forest=forest, messages=normalized, tools=[], chat_template_kwargs={},
            )
            if turn:
                assert parent == last_hash
            if delta:
                parent = forest.append_obs(
                    parent_state_hash=parent, rollout_id=0, abort_count=0,
                    messages_delta=delta, train_token_delta=[10 + turn],
                    rollout_token_delta=[10 + turn], tools=[] if turn == 0 else None,
                    chat_template_kwargs={} if turn == 0 else None,
                ).state_hash
            response = {"role": "assistant", "content": f"Policy reasoning {turn}."}
            last_hash = forest.append_resp(
                parent_state_hash=parent, rollout_id=0, abort_count=0,
                messages_delta=[response], train_token_delta=[100 + turn],
                rollout_token_delta=[100 + turn], logprob_delta=[-0.1],
            ).state_hash
            wire.record(sent, {"choices": [{"message": response}]})
            messages.append(AssistantMsg(name="audit", content=response["content"]))
            if turn == 0:
                messages.append(source[1])
            else:
                messages.append(UserMsg(name="user", content=f"Continue {turn}."))
        assert wire.rejected_turns == 0
        leaf = shard._implicit_export_state_hash(None, SimpleNamespace(forest=forest))
        sample = forest.build_sample(leaf_state_hash=leaf, tokenizer=SimpleNamespace(decode=lambda ids, **kw: str(ids)))
        assert sample.tokens == [10, 100, 11, 101, 12, 102]
        assert sample.rollout_tokens == sample.tokens
        assert sample.loss_mask == [1, 0, 1, 0, 1]
        assert sample.rollout_log_probs == [-0.1, 0.0, -0.1, 0.0, -0.1]
        assert sample.metadata["rollout_turns"] == 3
        assert sample.multimodal_inputs["images"] == ["https://example.com/1.png", "https://example.com/2.png"]
        forest.append_resp(
            parent_state_hash=parent, rollout_id=0, abort_count=0,
            messages_delta=[{"role": "assistant", "content": "A genuinely different branch."}],
            train_token_delta=[999], rollout_token_delta=[999], logprob_delta=[-0.2],
        )
        with pytest.raises(Exception, match="multiple_exportable_leaves"):
            shard._implicit_export_state_hash(None, SimpleNamespace(forest=forest))
    asyncio.run(run())
