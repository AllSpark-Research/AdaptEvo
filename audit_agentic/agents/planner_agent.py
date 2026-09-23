"""PlannerAgent: rule-aware label filtering + tool call planning."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..schemas import (
    AuditInput,
    FilteredLabel,
    PlannerOutput,
    RuleInfo,
    ToolCall,
)
from ..structured_output import planner_response_format, structured_output_enabled
from .base import BaseAgent
from .multimodal import DEFAULT_IMAGE_MAX_TOKENS, collect_rule_images


_ALLOWED_OPINIONS = {"support", "not_support", "uncertain"}
class PlannerAgent(BaseAgent):
    def __init__(self, llm_client, prompt_loader):
        super().__init__(llm_client, prompt_loader, name="planner")

    def run_plan(
        self,
        audit_input: AuditInput,
        shortlist_labels: List[str],
        shortlist_rules: List[RuleInfo],
        available_tools: List[Dict[str, Any]],
        image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
    ) -> tuple[PlannerOutput, Dict[str, Any]]:
        from ..environment.audit_experience import (
            load_audit_experience,
            planner_experience_enabled,
        )

        use_schema = structured_output_enabled(self.llm)
        default_template = "planner_schema" if use_schema else "planner"
        planner_template = os.environ.get("AUDIT_PLANNER_TEMPLATE", default_template)
        use_planner_experience = planner_experience_enabled(audit_input.source_id)
        audit_experience = (
            load_audit_experience(
                audit_input.source_id,
                include_all_labels=True,
            )
            if use_planner_experience
            else ""
        )
        tool_names_for_schema = [
            str(t.get("name") or t.get("tool_name") or "")
            for t in available_tools
        ]
        parsed, trace = self.run_template(
            planner_template,
            role_tag="planner",
            images=list(audit_input.images or []) + collect_rule_images(shortlist_rules),
            image_max_tokens=image_max_tokens,
            response_format=(
                planner_response_format(shortlist_labels, tool_names_for_schema)
                if use_schema
                else None
            ),
            note=audit_input.note,
            shortlist_labels=shortlist_labels,
            shortlist_rules=[r.model_dump() for r in shortlist_rules],
            available_tools=available_tools,
            audit_experience=audit_experience,
        )

        shortlist_set = set(shortlist_labels)
        tool_names = {t.get("name") or t.get("tool_name") for t in available_tools}
        tool_names.discard(None)

        # Sanitize filtered_labels
        def _coerce_filtered(entry: dict, *, allow_tools: bool) -> FilteredLabel | None:
            if not isinstance(entry, dict):
                return None
            label = entry.get("label")
            if not isinstance(label, str) or label not in shortlist_set:
                return None
            opinion = str(entry.get("preliminary_opinion", "uncertain")).strip().lower()
            if opinion not in _ALLOWED_OPINIONS:
                opinion = "uncertain"
            required = []
            tool_check = []
            if allow_tools:
                required = [
                    t for t in (entry.get("required_tools") or []) if t in tool_names
                ]
                tool_check = list(entry.get("tool_check_aspects") or [])
            return FilteredLabel(
                label=label,
                reason=str(entry.get("reason", "")),
                matched_post_evidence=list(entry.get("matched_post_evidence") or []),
                exemption_points=list(entry.get("exemption_points") or []),
                preliminary_opinion=opinion,
                required_tools=required,
                tool_check_aspects=tool_check,
            )

        filtered = parsed.get("filtered_labels") or {}
        possible_raw = filtered.get("possible_labels") or []
        tool_required_raw = filtered.get("tool_required_labels") or []

        possible_labels: List[FilteredLabel] = []
        for entry in possible_raw:
            cleaned = _coerce_filtered(entry, allow_tools=False)
            if cleaned is not None:
                possible_labels.append(cleaned)

        tool_required_labels: List[FilteredLabel] = []
        for entry in tool_required_raw:
            cleaned = _coerce_filtered(entry, allow_tools=True)
            if cleaned is not None:
                tool_required_labels.append(cleaned)

        # Enforce mutual exclusion at the Planner boundary. Tool-required
        # labels win because they carry the stronger evidence requirement.
        seen_labels = set()
        dedup_tool_required: List[FilteredLabel] = []
        for entry in tool_required_labels:
            if entry.label in seen_labels:
                continue
            seen_labels.add(entry.label)
            dedup_tool_required.append(entry)
        tool_required_labels = dedup_tool_required

        dedup_possible: List[FilteredLabel] = []
        for entry in possible_labels:
            if entry.label in seen_labels:
                continue
            seen_labels.add(entry.label)
            dedup_possible.append(entry)
        possible_labels = dedup_possible

        # Derive executable calls from tool_required_labels.required_tools.
        # This keeps every tool call attached to a concrete label requirement
        # and avoids a second, potentially inconsistent top-level tool list.
        calls: List[ToolCall] = []
        seen_tools = set()
        for label_entry in tool_required_labels:
            for name in label_entry.required_tools:
                if name not in tool_names or name in seen_tools:
                    continue
                seen_tools.add(name)
                calls.append(
                    ToolCall(
                        tool_name=name,
                        args={},
                        reason=f"required by label: {label_entry.label}",
                    )
                )

        trace_data = trace.model_dump()
        trace_data["audit_experience_enabled"] = use_planner_experience
        trace_data["audit_experience_chars"] = len(audit_experience)

        return (
            PlannerOutput(
                possible_labels=possible_labels,
                tool_required_labels=tool_required_labels,
                tool_calls=calls,
            ),
            trace_data,
        )
