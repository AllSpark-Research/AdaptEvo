"""MainAgent: two-stage agent (router + final)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..eval.parsing import parse_answer_labels
from ..schemas import AuditInput, MainFinalOutput, RouterOutput, RuleInfo
from ..structured_output import (
    build_final_candidate_labels,
    final_response_format,
    router_response_format,
    structured_output_enabled,
)
from .base import BaseAgent, _attach_llm_response_meta, _response_content
from .multimodal import collect_rule_images, make_multimodal_message, DEFAULT_IMAGE_MAX_TOKENS


def _label_from_entry(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("label", "")).strip()
    return str(getattr(entry, "label", "")).strip()


def _dedupe_planner_label_sections(
    possible_labels: List[Dict[str, Any]],
    tool_required_labels: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deduplicate labels before rendering main_final.

    If planner puts the same label in both buckets, keep the tool-required
    entry because it carries the stronger "needs tool evidence" signal.
    """
    seen = set()
    dedup_tool_required: List[Dict[str, Any]] = []
    for entry in tool_required_labels:
        label = _label_from_entry(entry)
        if not label or label in seen:
            continue
        dedup_tool_required.append(entry)
        seen.add(label)

    dedup_possible: List[Dict[str, Any]] = []
    for entry in possible_labels:
        label = _label_from_entry(entry)
        if not label or label in seen:
            continue
        dedup_possible.append(entry)
        seen.add(label)

    return dedup_possible, dedup_tool_required


class MainAgent(BaseAgent):
    def __init__(self, llm_client, prompt_loader):
        super().__init__(llm_client, prompt_loader, name="main")

    def run_router(self, audit_input: AuditInput, image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS) -> tuple[RouterOutput, Dict[str, Any]]:
        # Structured mode uses a dynamic candidate-label enum. Legacy mode
        # keeps the original <answer>label1,...</answer> parser.
        use_schema = structured_output_enabled(self.llm)
        default_template = "main_router_schema" if use_schema else "main_router"
        router_template = os.environ.get("AUDIT_MAIN_ROUTER_TEMPLATE", default_template)
        prompt = self.build_prompt(
            router_template,
            note=audit_input.note,
            candidate_labels=audit_input.candidate_labels,
        )
        messages = [make_multimodal_message("user", prompt, audit_input.images, image_max_tokens)]
        import time
        t0 = time.time()
        error: Optional[str] = None
        raw = ""
        labels: List[str] = []
        answer_text = ""
        parse_source = ""
        non_candidates: List[str] = []
        fuzzy: List[Dict[str, Any]] = []
        response: Any = None
        try:
            call_kwargs: Dict[str, Any] = {"role_tag": "main_router"}
            if use_schema:
                call_kwargs["response_format"] = router_response_format(
                    audit_input.candidate_labels
                )
            response = self.chat_with_meta(messages, **call_kwargs)
            raw = _response_content(response)
            if use_schema:
                parsed = self.parse_json_output(raw)
                answer_text = str(parsed.get("brief_analysis", ""))
                labels = list(parsed.get("shortlist_labels") or [])
                cand = set(audit_input.candidate_labels)
                non_candidates = [label for label in labels if label not in cand]
                labels = [label for label in labels if label in cand]
                labels = list(dict.fromkeys(labels))
                parse_source = "json_schema" if parsed else "json_parse_failed"
            else:
                labels, answer_text, parse_source, non_candidates, fuzzy = parse_answer_labels(
                    raw, audit_input.candidate_labels
                )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000.0

        # Record trace
        from ..schemas import AgentTrace
        trace = AgentTrace(
            role=self.name,
            template_name=router_template,
            rendered_prompt=prompt,
            raw_response=raw,
            parsed={
                "labels": labels,
                "answer_text": answer_text,
                "parse_source": parse_source,
                "non_candidate_labels": non_candidates,
                "fuzzy_mappings": fuzzy,
                "brief_analysis": answer_text,
                "structured_output": use_schema,
            },
            error=error,
            latency_ms=latency_ms,
        )
        _attach_llm_response_meta(trace, response)
        self.traces.append(trace)

        cand = set(audit_input.candidate_labels)
        shortlist = [l for l in labels if l in cand]
        output = RouterOutput(
            shortlist_labels=shortlist,
            related_labels=[],
            likely_pass=("通过" in shortlist and len(shortlist) == 1),
            uncertainty="medium",
            reason=answer_text or (raw[:200] if raw else ""),
        )
        return output, trace.model_dump()

    def run_final(
        self,
        audit_input: AuditInput,
        router_trace: Dict[str, Any],
        possible_labels: List[Dict[str, Any]],
        tool_required_labels: List[Dict[str, Any]],
        remaining_rules: List[RuleInfo],
        tool_observations: List[Dict[str, Any]],
        judge_feedback: Optional[Dict[str, Any]] = None,
        shortlist_labels: Optional[List[str]] = None,
        final_candidate_labels: Optional[List[str]] = None,
        standalone: bool = False,
        **kwargs: Any,
    ) -> tuple[MainFinalOutput, Dict[str, Any]]:
        """Run the final decision turn.

        Large queues continue the main_router conversation. Small queues use
        a standalone prompt containing the note, note images, Planner output,
        rules, and tool observations.
        """
        # Render tool observations into a single prompt section + aligned images.
        # Tool images are NOT part of audit_input.images; they ride along their
        # own <image> placeholders inside the continuation message.
        from ..environment.audit_experience import load_audit_experience
        from ..environment.tool_render import render_observations
        from ..environment.tool_registry import DEFAULT_TOOL_REGISTRY

        registry = getattr(self, "tool_registry", None) or DEFAULT_TOOL_REGISTRY
        tool_section, tool_images = render_observations(tool_observations, registry)
        rule_images = collect_rule_images(remaining_rules)
        possible_labels, tool_required_labels = _dedupe_planner_label_sections(
            possible_labels,
            tool_required_labels,
        )

        use_schema = structured_output_enabled(self.llm)
        default_template = "main_final_schema" if use_schema else "main_final"
        final_template = os.environ.get("AUDIT_MAIN_FINAL_TEMPLATE", default_template)
        shortlist_labels = shortlist_labels or audit_input.candidate_labels
        final_candidate_labels = final_candidate_labels or build_final_candidate_labels(
            audit_input.candidate_labels,
            shortlist_labels,
            possible_labels,
            tool_required_labels,
        )
        # Reviewer-summarized labeling heuristics, keyed by source (see
        # environment/cache/experience/audit_experience.json). Empty string
        # for any source without configured experience, so this is a no-op
        # unless AUDIT_MAIN_FINAL_TEMPLATE=main_final_new is set and the
        # source has an entry — other templates simply ignore the extra kwarg.
        audit_experience = load_audit_experience(
            audit_input.source_id, labels=final_candidate_labels
        )
        continuation_prompt = self.build_prompt(
            final_template,
            standalone_mode=standalone,
            note=audit_input.note,
            final_candidate_labels=final_candidate_labels,
            possible_labels=possible_labels,
            tool_required_labels=tool_required_labels,
            remaining_rules=[r.model_dump() for r in remaining_rules],
            tool_section=tool_section,
            judge_feedback=judge_feedback,
            audit_experience=audit_experience,
        )

        router_user_prompt = router_trace.get("rendered_prompt", "") if router_trace else ""
        router_assistant_response = router_trace.get("raw_response", "") if router_trace else ""

        image_max_tokens = kwargs.get("image_max_tokens", DEFAULT_IMAGE_MAX_TOKENS)
        if standalone:
            messages = [
                make_multimodal_message(
                    "user",
                    continuation_prompt,
                    list(audit_input.images or []) + rule_images + tool_images,
                    image_max_tokens,
                )
            ]
        else:
            messages = [
                make_multimodal_message("user", router_user_prompt, audit_input.images, image_max_tokens),
                {"role": "assistant", "content": router_assistant_response},
                make_multimodal_message("user", continuation_prompt, rule_images + tool_images, image_max_tokens),
            ]
        response_format = final_response_format(final_candidate_labels) if use_schema else None
        parsed, trace = self.run_messages(
            messages,
            role_tag="main_final",
            template_name=final_template,
            response_format=response_format,
        )
        # Some models drift back to the router-style <answer>...</answer> format
        # in the continuation turn. Salvage it so valid label decisions are not
        # counted as empty predictions in eval traces.
        if not parsed and trace.raw_response:
            labels, answer_text, parse_source, non_candidates, fuzzy = parse_answer_labels(
                trace.raw_response, audit_input.candidate_labels
            )
            if labels:
                parsed = {
                    "predict_label": labels,
                    "audit_trace": answer_text or trace.raw_response,
                }
                try:
                    trace.parsed = {
                        "fallback": "answer_tag",
                        "labels": labels,
                        "answer_text": answer_text,
                        "parse_source": parse_source,
                        "non_candidate_labels": non_candidates,
                        "fuzzy_mappings": fuzzy,
                    }
                    trace.error = None
                except Exception:
                    pass

        cand = set(final_candidate_labels)
        predict = [x for x in parsed.get("predict_label", []) if x in cand]
        binary = str(parsed.get("binary_decision", "pass")).strip().lower()
        if binary not in {"pass", "violation"}:
            binary = "violation" if predict and not _all_are_pass(predict) else "pass"

        output = MainFinalOutput(
            predict_label=predict,
            binary_decision=binary,
            audit_trace=str(parsed.get("audit_trace", "")),
            used_rules=list(parsed.get("used_rules", []) or []),
            used_tools=list(parsed.get("used_tools", []) or []),
        )
        # Keep the effective experience configuration in the trace. Without
        # these fields an eval can silently fall back to main_final_schema,
        # which accepts the audit_experience kwarg but does not render it.
        trace_data = trace.model_dump()
        trace_data["template_name"] = final_template
        trace_data["audit_experience_enabled"] = bool(audit_experience)
        trace_data["audit_experience_chars"] = len(audit_experience)
        return output, trace_data


def _all_are_pass(labels: List[str]) -> bool:
    return all("通过" in l or l.lower() in {"pass", "通过"} for l in labels)
