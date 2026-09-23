"""AuditWorkflow: orchestrates the full router -> rules -> planner -> tools
-> main_final -> (optional judge -> main_final revision) loop.

The workflow returns an AuditWorkflowResult that contains everything needed
to:

* expose to inference callers (predict_label + trace),
* turn into RL training samples (each agent's prompt + response are kept in
  workflow_trace under stable keys).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents.judge_agent import JudgeAgent
from .agents.main_agent import MainAgent
from .agents.planner_agent import PlannerAgent
from .environment.rule_retriever import BaseRuleRetriever, QUEUE_NOTICE_LABEL, retrieve_rules_with_id_fallback
from .environment.tool_executor import ToolExecutor
from .environment.tool_registry import list_tool_briefs
from .schemas import (
    AuditInput,
    AuditWorkflowResult,
    JudgeOutput,
    MainFinalOutput,
    PlannerOutput,
    RouterOutput,
    RuleInfo,
    ToolObservation,
)
from .structured_output import build_final_candidate_labels, router_bypass_max_labels


class AuditWorkflow:
    def __init__(
        self,
        main_agent: MainAgent,
        planner_agent: PlannerAgent,
        rule_retriever: BaseRuleRetriever,
        tool_executor: ToolExecutor,
        judge_agent: Optional[JudgeAgent] = None,
        enable_judge: bool = False,
        max_revision_rounds: int = 1,
    ):
        self.main_agent = main_agent
        self.planner_agent = planner_agent
        self.rule_retriever = rule_retriever
        self.tool_executor = tool_executor
        self.judge_agent = judge_agent
        self.enable_judge = enable_judge and judge_agent is not None
        self.max_revision_rounds = max(0, int(max_revision_rounds))

    # --- main entry ----------------------------------------------------------

    def run(self, audit_input: AuditInput) -> AuditWorkflowResult:
        trace: Dict[str, Any] = {}
        errors: List[str] = []

        # 1. Conditional router. Small candidate sets keep every label without
        # spending an LLM call or creating a synthetic assistant turn.
        router_invoked = len(audit_input.candidate_labels) > router_bypass_max_labels()
        if router_invoked:
            router_out, router_trace = self._safe(
                lambda: self.main_agent.run_router(audit_input),
                errors,
                "router",
                default=(RouterOutput(), {}),
            )
            selection_mode = "llm_router"
        else:
            router_out = RouterOutput(shortlist_labels=list(audit_input.candidate_labels))
            router_trace = {}
            selection_mode = "all_candidates"
        trace["router"] = {
            "output": router_out.model_dump(),
            "agent_trace": router_trace,
            "router_invoked": router_invoked,
            "selection_mode": selection_mode,
        }

        # 2. retrieve rules for shortlist labels (+pass label if present)
        labels_for_rules = list(router_out.shortlist_labels)
        if not labels_for_rules:
            labels_for_rules = list(audit_input.candidate_labels)
        rules = self._safe(
            lambda: retrieve_rules_with_id_fallback(self.rule_retriever, audit_input, labels_for_rules),
            errors,
            "rule_retrieve",
            default=[],
        )
        trace["retrieved_rules"] = [r.model_dump() for r in rules]

        # 3. planner
        available_tools = list_tool_briefs()
        # Optional: filter to user-allowed tools if audit_input.available_tools given
        if audit_input.available_tools:
            allowed = set(audit_input.available_tools)
            available_tools = [t for t in available_tools if t["name"] in allowed]

        planner_out, planner_trace = self._safe(
            lambda: self.planner_agent.run_plan(
                audit_input,
                shortlist_labels=labels_for_rules,
                shortlist_rules=rules,
                available_tools=available_tools,
            ),
            errors,
            "planner",
            default=(PlannerOutput(), {}),
        )
        trace["planner"] = {"output": planner_out.model_dump(), "agent_trace": planner_trace}

        # 4. tools — auto-inject current note_id into every tool_call
        bound_executor = self.tool_executor.with_defaults(
            note_id=audit_input.note_id,
            source_id=audit_input.source_id,
        )
        observations: List[ToolObservation] = self._safe(
            lambda: bound_executor.execute(planner_out.tool_calls),
            errors,
            "tool_executor",
            default=[],
        )
        trace["tool_observations"] = [o.model_dump() for o in observations]

        # 5. main final (round 0) — continuation of router turn
        # Only feed rules for labels that survived planner filter (rule_out 的不再喂)
        final_candidate_labels = build_final_candidate_labels(
            audit_input.candidate_labels,
            labels_for_rules,
            planner_out.possible_labels,
            planner_out.tool_required_labels,
        )
        surviving_labels = set(final_candidate_labels)
        remaining_rules = [r for r in rules if r.label in surviving_labels or r.label == QUEUE_NOTICE_LABEL]
        router_trace = trace["router"]["agent_trace"]

        main_out, main_trace = self._safe(
            lambda: self.main_agent.run_final(
                audit_input,
                router_trace=router_trace,
                possible_labels=[fl.model_dump() for fl in planner_out.possible_labels],
                tool_required_labels=[fl.model_dump() for fl in planner_out.tool_required_labels],
                remaining_rules=remaining_rules,
                tool_observations=[o.model_dump() for o in observations],
                judge_feedback=None,
                shortlist_labels=labels_for_rules,
                final_candidate_labels=final_candidate_labels,
                standalone=not router_invoked,
            ),
            errors,
            "main_final",
            default=(MainFinalOutput(), {}),
        )
        trace["main_final_initial"] = {"output": main_out.model_dump(), "agent_trace": main_trace}

        # 6. optional judge + revision
        judge_out_final: Optional[JudgeOutput] = None
        if self.enable_judge:
            for round_idx in range(self.max_revision_rounds + 1):
                judge_out, judge_trace = self._safe(
                    lambda: self.judge_agent.run_verify(
                        audit_input,
                        shortlist_rules=rules,
                        tool_observations=[o.model_dump() for o in observations],
                        main_output=main_out,
                    ),
                    errors,
                    f"judge_round_{round_idx}",
                    default=(JudgeOutput(), {}),
                )
                trace.setdefault("judge", []).append(
                    {"output": judge_out.model_dump(), "agent_trace": judge_trace}
                )
                judge_out_final = judge_out
                if judge_out.verdict == "pass":
                    break
                if round_idx >= self.max_revision_rounds:
                    break
                # Revise
                revised, revised_trace = self._safe(
                    lambda: self.main_agent.run_final(
                        audit_input,
                        router_trace=router_trace,
                        possible_labels=[fl.model_dump() for fl in planner_out.possible_labels],
                        tool_required_labels=[fl.model_dump() for fl in planner_out.tool_required_labels],
                        remaining_rules=remaining_rules,
                        tool_observations=[o.model_dump() for o in observations],
                        judge_feedback=judge_out.model_dump(),
                        shortlist_labels=labels_for_rules,
                        final_candidate_labels=final_candidate_labels,
                        standalone=not router_invoked,
                    ),
                    errors,
                    f"main_final_revision_{round_idx}",
                    default=(main_out, {}),
                )
                main_out = revised
                trace.setdefault("main_final_revisions", []).append(
                    {"output": revised.model_dump(), "agent_trace": revised_trace}
                )

        trace["main_final_resolved"] = main_out.model_dump()

        result = AuditWorkflowResult(
            note_id=audit_input.note_id,
            source_id=audit_input.source_id,
            predict_label=main_out.predict_label,
            binary_decision=main_out.binary_decision,
            audit_trace_text=main_out.audit_trace,
            used_rules=main_out.used_rules,
            used_tools=main_out.used_tools,
            planner_analysis=planner_out.model_dump(),
            workflow_trace=trace,
            errors=errors,
        )
        return result

    # --- helper --------------------------------------------------------------

    @staticmethod
    def _safe(fn, errors_sink, step_name, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            errors_sink.append(f"{step_name}: {type(exc).__name__}: {exc}")
            return default
