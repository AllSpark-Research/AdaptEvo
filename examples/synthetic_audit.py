"""Run one synthetic case through the native AgentScope tool loop."""

import asyncio
import json
import os

from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel

from audit_agentic.agents.agentscope_audit_agent import AgentScopeAuditAgent
from audit_agentic.environment.tool_executor import ToolExecutor
from audit_agentic.environment.tools.base_tool import BaseTool, ToolRenderResult
from audit_agentic.prompts.loader import default_loader
from audit_agentic.schemas import AuditInput, RuleInfo


class LookupSyntheticEvidence(BaseTool):
    name = "lookup_synthetic_evidence"
    description = "Return the fixed evidence for this synthetic example."
    brief = description
    public_input_schema = {
        "type": "object", "properties": {}, "additionalProperties": False,
    }

    def run(self, args):
        return {"verified": True, "statement": "This item is explicitly marked as a fictional example."}

    def render(self, result):
        return ToolRenderResult(text=json.dumps(result), images=[])


async def main():
    os.environ["AUDIT_DISABLE_ONLINE_SERVICES"] = "1"
    tool = LookupSyntheticEvidence()
    registry = {tool.name: tool}
    model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ["OPENAI_BASE_URL"],
        ),
        model=os.environ["OPENAI_MODEL"],
        parameters=OpenAIChatModel.Parameters(temperature=0.0, max_tokens=2048),
        stream=False,
        max_retries=1,
        client_kwargs={"timeout": 120},
    )
    agent = AgentScopeAuditAgent(
        model=model,
        prompt_loader=default_loader(),
        tool_executor=ToolExecutor(registry),
        tool_registry=registry,
        experience_enabled=False,
        compression_enabled=False,
        max_iters=4,
        full_trace=False,
    )
    case = AuditInput.from_data_row({
        "note": "A fictional item used only to demonstrate a moderation workflow.",
        "images": [],
        "metadata": {"note_id": "synthetic-001", "source": "synthetic"},
    })
    result, trace = await agent.run(
        case,
        candidate_labels=["通过", "DEMO_UNVERIFIED"],
        rules=[RuleInfo(
            label="DEMO_UNVERIFIED",
            rule_text=("Synthetic rule: use lookup_synthetic_evidence to verify this item. "
                       "If verified=true, return 通过. Otherwise return DEMO_UNVERIFIED. "
                       "This rule is only for this example, not a production policy."),
            required_tools=[tool.name],
        )],
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
