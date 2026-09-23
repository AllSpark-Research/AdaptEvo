"""JudgeAgent: optional verifier of main agent output."""

from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import AuditInput, JudgeOutput, MainFinalOutput, RuleInfo
from .base import BaseAgent
from .multimodal import DEFAULT_IMAGE_MAX_TOKENS, collect_rule_images


_ALLOWED_ISSUES = {
    "rule_mismatch",
    "insufficient_evidence",
    "missing_tool",
    "tool_misuse",
    "missing_exemption_check",
    "label_confusion",
    "hallucination",
    "invalid_format",
}


class JudgeAgent(BaseAgent):
    def __init__(self, llm_client, prompt_loader):
        super().__init__(llm_client, prompt_loader, name="judge")

    def run_verify(
        self,
        audit_input: AuditInput,
        shortlist_rules: List[RuleInfo],
        tool_observations: List[Dict[str, Any]],
        main_output: MainFinalOutput,
        image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
    ) -> tuple[JudgeOutput, Dict[str, Any]]:
        from ..environment.tool_render import render_observations
        from ..environment.tool_registry import DEFAULT_TOOL_REGISTRY

        tool_section, tool_images = render_observations(tool_observations, DEFAULT_TOOL_REGISTRY)
        parsed, trace = self.run_template(
            "judge_verifier",
            role_tag="judge",
            images=list(audit_input.images or []) + collect_rule_images(shortlist_rules) + tool_images,
            image_max_tokens=image_max_tokens,
            note=audit_input.note,
            shortlist_rules=[r.model_dump() for r in shortlist_rules],
            tool_section=tool_section,
            main_agent_output=main_output.model_dump(),
        )

        verdict = str(parsed.get("verdict", "pass")).strip().lower()
        if verdict not in {"pass", "fail"}:
            verdict = "pass"
        issues = [i for i in (parsed.get("issue_types") or []) if i in _ALLOWED_ISSUES]
        return (
            JudgeOutput(
                verdict=verdict,
                issue_types=issues,
                critique=str(parsed.get("critique", "")),
                revision_instruction=str(parsed.get("revision_instruction", "")),
            ),
            trace.model_dump(),
        )
