from __future__ import annotations

import pytest

from audit_agentic.rewards.reward_adaptive import (
    compute_final_reward,
    human_agreement_confidence,
    outcome_reward,
    process_reward,
    rule_support_gate,
)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["A", "A", "A", "A"], 1.0),
        (["A", "A", "A", "B"], 0.5943609377704335),
        (["A", "A", "B", "B"], 0.5),
        (["A", "A", "B", "C"], 0.25),
        (["A", "B", "C", "D"], 0.0),
    ],
)
def test_human_agreement_confidence(labels, expected) -> None:
    assert human_agreement_confidence(labels) == pytest.approx(expected)


def test_rule_support_gate_mapping_and_missing_default() -> None:
    assert rule_support_gate("supported") == 1.0
    assert rule_support_gate({"verdict": "ambiguous"}) == 0.6
    assert rule_support_gate("unsupported") == 0.2
    assert rule_support_gate(None) == 0.6


def test_hierarchical_outcome_reward() -> None:
    assert outcome_reward("违规A", "违规A") == 1.0
    assert outcome_reward("违规A", "违规B") == 0.6
    assert outcome_reward("违规A", "通过") == 0.0
    assert outcome_reward("通过", "通过") == 1.0
    assert outcome_reward("通过", "违规A") == 0.0
    assert outcome_reward("0|通过", "通过") == 1.0


def test_process_reward_uses_strict_judge_scores() -> None:
    assert process_reward(
        {
            "factual_grounding": 1.0,
            "rule_fidelity": 0.8,
            "evidence_coverage": 0.6,
            "tool_use": 0.4,
            "decision_consistency": 0.2,
        }
    ) == pytest.approx(0.66)
    assert process_reward(None) is None


def test_missing_process_reward_leaves_process_component_empty() -> None:
    result = compute_final_reward(
        human_labels=["A", "A", "A", "A"],
        rule_verdict="supported",
        gt="A",
        prediction="A",
    )
    assert result["gt_confidence"] == 1.0
    assert result["outcome_weight"] == pytest.approx(0.9)
    assert result["process_weight"] == pytest.approx(0.1)
    assert result["process_reward_available"] == 0.0
    assert result["process_component"] == 0.0
    assert result["final_reward"] == pytest.approx(0.9)
    assert result["advantage_weight"] == pytest.approx(0.9)


def test_process_reward_fills_adaptive_mixture() -> None:
    result = compute_final_reward(
        human_labels=["A", "A", "A", "A"],
        rule_verdict="ambiguous",
        gt="A",
        prediction="A",
        process_reward_value=0.5,
    )
    assert result["outcome_weight"] == pytest.approx(0.54)
    assert result["process_weight"] == pytest.approx(0.46)
    assert result["final_reward"] == pytest.approx(0.77)


@pytest.mark.parametrize(
    ("labels", "expected_confidence", "expected_weight"),
    [
        (["A", "A", "A", "A"], 1.0, 0.9),
        (["A", "A", "A", "B"], 0.5943609377704335, 0.7374664266447784),
        (["A", "A", "B", "B"], 0.5, 0.7060660171779821),
        (["A", "A", "B", "C"], 0.25, 0.6375),
        (["A", "B", "C", "D"], 0.0, 0.6),
    ],
)
def test_bounded_gt_weight_curve(
    labels,
    expected_confidence,
    expected_weight,
) -> None:
    result = compute_final_reward(
        human_labels=labels,
        rule_verdict="supported",
        gt="A",
        prediction="A",
        process_reward_value=0.5,
        gamma=1.5,
        gt_weight_min=0.6,
        gt_weight_max=0.9,
    )
    assert result["gt_confidence"] == pytest.approx(expected_confidence)
    assert result["outcome_weight"] == pytest.approx(expected_weight)
    assert result["process_weight"] == pytest.approx(1.0 - expected_weight)
    assert result["bounded_weight_enabled"] == 1.0
    assert result["gt_weight_min"] == 0.6
    assert result["gt_weight_max"] == 0.9


def test_bounded_gt_weight_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="GT weight bounds"):
        compute_final_reward(
            human_labels=["A", "A", "A", "A"],
            rule_verdict="supported",
            gt="A",
            prediction="A",
            gt_weight_min=0.9,
            gt_weight_max=0.6,
        )


def test_format_penalty_matches_current_agentscope_semantics() -> None:
    one_error = compute_final_reward(
        human_labels=["A", "A", "A", "A"],
        rule_verdict="supported",
        gt="A",
        prediction="A",
        tool_name_format_error_count=1,
    )
    many_errors = compute_final_reward(
        human_labels=["A", "A", "A", "A"],
        rule_verdict="supported",
        gt="A",
        prediction="A",
        unknown_tool_call_count=9,
    )
    assert one_error["format_penalty"] == pytest.approx(0.1)
    assert one_error["final_reward"] == pytest.approx(0.8)
    assert many_errors["format_penalty"] == pytest.approx(0.3)
    assert many_errors["final_reward"] == pytest.approx(0.6)


def test_invalid_final_output_is_exactly_zero() -> None:
    result = compute_final_reward(
        human_labels=["A", "A", "A", "A"],
        rule_verdict="supported",
        gt="A",
        prediction="A",
        process_reward_value=1.0,
        invalid_final_output=True,
        unknown_tool_call_count=3,
    )
    assert result["format_penalty"] == pytest.approx(0.3)
    assert result["final_reward"] == 0.0
