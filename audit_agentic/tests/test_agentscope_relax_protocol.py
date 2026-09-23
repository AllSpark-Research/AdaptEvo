from __future__ import annotations

from audit_agentic.agents.agentscope_audit_agent import (
    parse_final_structured_output,
)
from audit_agentic.relax_app.protocol_openai_model import (
    ProtocolAwareOpenAIChatModel,
)
from audit_agentic.relax_app.protocol_reward import score_agentscope_protocol
from audit_agentic.relax_app.relax_wire_history import RelaxWireHistory
from audit_agentic.relax_app.agentscope_agent import _is_infrastructure_exception


TOOLS = [
    {"type": "function", "function": {"name": "get_detail_rule"}},
    {"type": "function", "function": {"name": "get_comments"}},
]


def _model() -> ProtocolAwareOpenAIChatModel:
    model = object.__new__(ProtocolAwareOpenAIChatModel)
    model._turn_index = 1
    model._protocol_events = []
    model._wire_history = RelaxWireHistory()
    return model


def _tool_response(name: str, arguments: str = "{}") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        },
                    ],
                },
            },
        ],
    }


def test_invalid_or_missing_final_label_is_diagnostic_not_exception() -> None:
    structured, label, errors = parse_final_structured_output(
        {"predict_label": "not-a-candidate"},
        ["通过", "违规标签"],
    )
    assert structured == {"predict_label": "not-a-candidate"}
    assert label == "not-a-candidate"
    assert errors == ["invalid_predict_label"]

    structured, label, errors = parse_final_structured_output(None, ["通过"])
    assert structured == {}
    assert label == ""
    assert errors == [
        "missing_or_malformed_structured_output",
        "missing_predict_label",
    ]


def test_invalid_final_output_gets_exactly_zero_reward() -> None:
    result = score_agentscope_protocol(
        prediction=[],
        gt_labels=["违规标签"],
        invalid_final_output=True,
        final_protocol_error_count=1,
    )
    assert result["score"] == 0.0
    assert result["classification_score"] == 0.0
    assert result["nOA"] == 0.0
    assert result["wOA"] == 0.0


def test_tool_protocol_penalty_is_point_one_and_capped_at_point_three() -> None:
    one_error = score_agentscope_protocol(
        prediction=["通过"],
        gt_labels=["通过"],
        invalid_final_output=False,
        tool_name_format_error_count=1,
    )
    many_errors = score_agentscope_protocol(
        prediction=["通过"],
        gt_labels=["通过"],
        invalid_final_output=False,
        unknown_tool_call_count=9,
    )
    assert one_error["score"] == 0.9
    assert one_error["tool_protocol_penalty"] == 0.1
    assert many_errors["score"] == 0.7
    assert many_errors["tool_protocol_penalty"] == 0.3


def test_valid_multi_gt_keeps_existing_overlap_reward_semantics() -> None:
    result = score_agentscope_protocol(
        prediction=["标签A"],
        gt_labels=["标签A", "标签B"],
        invalid_final_output=False,
    )
    assert result["score"] > 0.0
    assert result["overlap"] == 1.0


def test_tool_name_repair_is_recorded_and_reused_on_wire_history() -> None:
    model = _model()
    response = _tool_response("GetDetailRule")
    repairs = model._record_protocol_events(response, TOOLS)
    model._wire_history.record(
        [{"role": "user", "content": "audit"}],
        response,
    )
    model._apply_tool_name_repairs(response, repairs)

    assert repairs == {(0, 0): "get_detail_rule"}
    assert model.protocol_diagnostics["tool_name_format_error_count"] == 1
    assert (
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_detail_rule"
    )

    rebuilt = [
        {"role": "user", "content": "audit"},
        response["choices"][0]["message"],
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "rule detail",
        },
    ]
    reconciled = model._wire_history.reconcile(rebuilt)
    assert (
        reconciled[1]["tool_calls"][0]["function"]["name"]
        == "GetDetailRule"
    )
    assert reconciled[2]["content"] == "rule detail"
    assert model._wire_history.canonicalized_turns == 1


def test_unrelated_tool_is_not_silently_repaired() -> None:
    model = _model()
    response = _tool_response("example_queue_002")
    repairs = model._record_protocol_events(response, TOOLS)
    assert repairs == {}
    assert model.protocol_diagnostics["unknown_tool_call_count"] == 1


def test_infrastructure_errors_are_not_converted_to_zero_reward() -> None:
    assert _is_infrastructure_exception(
        RuntimeError("503 Service Unavailable from Relax proxy"),
    )
    assert not _is_infrastructure_exception(
        RuntimeError("Tool 'GetDetailRule' is not defined in the tools list."),
    )
