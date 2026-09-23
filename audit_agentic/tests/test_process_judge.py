import json

import pytest

from audit_agentic.rewards.process_judge import (
    PROCESS_JUDGE_RESPONSE_FORMAT,
    ProcessJudgeOutputError,
    normalize_process_judge_output,
)


def _payload(**updates):
    payload = {
        "factual_grounding": 0.8,
        "rule_fidelity": 0.6,
        "evidence_coverage": 0.5,
        "tool_use": 1.0,
        "decision_consistency": 0.7,
    }
    payload.update(updates)
    return payload


def test_computes_weighted_process_reward():
    result = normalize_process_judge_output(json.dumps(_payload()))
    assert result["process_reward"] == pytest.approx(0.705)
    assert set(result) == {
        "factual_grounding",
        "rule_fidelity",
        "evidence_coverage",
        "tool_use",
        "decision_consistency",
        "process_reward",
    }


def test_rejects_out_of_range_score():
    with pytest.raises(ProcessJudgeOutputError, match="factual_grounding"):
        normalize_process_judge_output(_payload(factual_grounding=1.2))


def test_ignores_extra_fields_from_legacy_output():
    result = normalize_process_judge_output(
        _payload(brief_reason="legacy", process_reward=0.0),
    )
    assert result["process_reward"] == pytest.approx(0.705)


def test_strict_mode_rejects_extra_fields():
    with pytest.raises(ProcessJudgeOutputError, match="strict schema"):
        normalize_process_judge_output(
            _payload(brief_reason="not allowed"),
            strict=True,
        )


def test_response_format_is_strict_json_schema():
    schema = PROCESS_JUDGE_RESPONSE_FORMAT["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) == set(_payload())
