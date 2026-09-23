"""Data schemas for the audit agentic workflow.

All schemas are pydantic models so they can be JSON-serialised. Fields are
intentionally permissive (extra='allow', most lists/dicts optional) so that the
workflow can carry real-world data without strict up-front contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _PermissiveModel(BaseModel):
    """Base model that allows extra fields - real data has more keys than we predeclare."""

    model_config = ConfigDict(extra="allow")


def _strip_label_id(label: str) -> str:
    """Strip numeric id prefix from a label string.

    "50005495|竞对平台导流" → "竞对平台导流"
    "通过" → "通过"  (no change, backward-compat)
    """
    if "|" in label:
        return label.split("|", 1)[1]
    return label


class AuditInput(_PermissiveModel):
    """Single audit task input.

    Matches the cleaned data shape: ``{note, images, metadata}`` where
    ``metadata`` contains ``note_id, user_id, source, original_source, dtm,
    candidate_labels, labels``. Use :py:meth:`from_data_row` to build one
    directly from a jsonl row.
    """

    # Core fields aligned with cleaned jsonl
    note: str = ""
    images: List[str] = Field(default_factory=list)

    # Identifiers + dispatch keys (sourced from row['metadata'])
    note_id: Optional[str] = None
    history_id: Optional[str] = None
    source_id: str = ""
    candidate_labels: List[str] = Field(default_factory=list)
    gt_labels: Optional[List[str]] = None

    # Raw labels with id prefix (e.g. "50005495|竞对平台导流") preserved for
    # reward computation / debugging. candidate_labels / gt_labels always hold
    # the stripped name-only form that is shown to the model and used for rule
    # retrieval.
    raw_candidate_labels: List[str] = Field(default_factory=list)
    raw_gt_labels: Optional[List[str]] = None

    # Optional ancillary
    label_cards: Dict[str, str] = Field(default_factory=dict)
    available_tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_data_row(cls, row: Dict[str, Any]) -> "AuditInput":
        """Construct from either format:

        1. Original cleaned format (top-level ``note``):
           ``{"note": "...", "images": [...], "metadata": {...}}``

        2. Relax training format (note inside ``metadata.audit_input``):
           ``{"prompt": [...], "label": "...", "images": [...],
             "metadata": {..., "audit_input": {"note": "...", ...}}}``

        Both are auto-detected; fields in ``audit_input`` take precedence
        when present (they are the canonical structured form).
        """
        md = row.get("metadata") or {}
        ai = md.get("audit_input") or {}

        # note: prefer audit_input.note > top-level note
        note = ai.get("note") or row.get("note", "")
        images = list(ai.get("images") or row.get("images") or [])
        note_id = ai.get("note_id") or md.get("note_id")
        history_id = ai.get("history_id") or md.get("history_id")
        source_id = ai.get("source_id") or md.get("source", "") or ""
        raw_candidate_labels = list(
            ai.get("raw_candidate_labels")
            or ai.get("candidate_labels")
            or md.get("candidate_labels")
            or []
        )
        raw_gt = ai.get("raw_gt_labels") or ai.get("gt_labels") or md.get("labels")
        raw_gt_labels = list(raw_gt) if raw_gt is not None else None
        label_cards = dict(ai.get("label_cards") or md.get("label_cards") or {})
        available_tools = list(ai.get("available_tools") or md.get("available_tools") or [])

        # Strip numeric id prefix ("50005495|竞对平台导流" → "竞对平台导流").
        # Labels without "|" are kept as-is (backward-compat with old data).
        candidate_labels = [_strip_label_id(l) for l in raw_candidate_labels]
        gt_labels = [_strip_label_id(l) for l in raw_gt_labels] if raw_gt_labels is not None else None

        return cls(
            note=note,
            images=images,
            note_id=note_id,
            history_id=history_id,
            source_id=source_id,
            candidate_labels=candidate_labels,
            gt_labels=gt_labels,
            raw_candidate_labels=raw_candidate_labels,
            raw_gt_labels=raw_gt_labels,
            label_cards=label_cards,
            available_tools=available_tools,
            metadata=md,
        )


class RouterOutput(_PermissiveModel):
    shortlist_labels: List[str] = Field(default_factory=list)
    related_labels: List[str] = Field(default_factory=list)
    likely_pass: bool = False
    uncertainty: str = "medium"
    reason: str = ""


class RuleInfo(_PermissiveModel):
    """Container for retrieved rule info for one label.

    The simplest path (V1): ``rule_text`` is the full markdown rule body
    rendered by the retriever and injected directly into the Planner /
    Main-final prompts. The structured ``positive_conditions`` etc. fields
    remain optional for retrievers that want to expose them.
    """

    label: str
    rule_text: str = ""
    rule_images: List[str] = Field(default_factory=list)
    positive_conditions: List[str] = Field(default_factory=list)
    exemption_conditions: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    hard_negatives: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(_PermissiveModel):
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolObservation(_PermissiveModel):
    tool_name: str
    status: str = "ok"
    result: Any = None
    error: Optional[str] = None
    call: Optional[ToolCall] = None


class FilteredLabel(_PermissiveModel):
    """A single entry in planner's filtered_labels (possible or tool_required)."""

    label: str
    reason: str = ""
    matched_post_evidence: List[str] = Field(default_factory=list)
    exemption_points: List[str] = Field(default_factory=list)
    preliminary_opinion: str = "uncertain"
    # tool_required-specific (empty for possible_labels)
    required_tools: List[str] = Field(default_factory=list)
    tool_check_aspects: List[str] = Field(default_factory=list)


class PlannerOutput(_PermissiveModel):
    possible_labels: List[FilteredLabel] = Field(default_factory=list)
    tool_required_labels: List[FilteredLabel] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)


class MainFinalOutput(_PermissiveModel):
    predict_label: List[str] = Field(default_factory=list)
    binary_decision: str = "pass"
    audit_trace: str = ""
    used_rules: List[str] = Field(default_factory=list)
    used_tools: List[str] = Field(default_factory=list)


class JudgeOutput(_PermissiveModel):
    verdict: str = "pass"
    issue_types: List[str] = Field(default_factory=list)
    critique: str = ""
    revision_instruction: str = ""


class AgentTrace(_PermissiveModel):
    """One LLM call trace: prompt sent, raw response, parsed output."""

    role: str
    template_name: str
    rendered_prompt: str = ""
    raw_response: str = ""
    parsed: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class AuditWorkflowResult(_PermissiveModel):
    note_id: Optional[str] = None
    source_id: str
    predict_label: List[str] = Field(default_factory=list)
    binary_decision: str = "pass"
    audit_trace_text: str = ""
    used_rules: List[str] = Field(default_factory=list)
    used_tools: List[str] = Field(default_factory=list)
    planner_analysis: Dict[str, Any] = Field(default_factory=dict)
    workflow_trace: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
