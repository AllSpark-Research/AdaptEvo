import json

from audit_agentic.agents.llm_client import MockLLMClient
from audit_agentic.agents.planner_agent import PlannerAgent
from audit_agentic.prompts.loader import default_loader
from audit_agentic.schemas import AuditInput
from audit_agentic.structured_output import (
    build_final_candidate_labels,
    final_output_quality,
    final_response_format,
    planner_output_quality,
    planner_response_format,
    router_output_quality,
    router_response_format,
)


def test_router_schema_uses_candidate_enum_and_range():
    labels = [f"标签{i}" for i in range(20)]
    schema = router_response_format(labels)["json_schema"]["schema"]
    shortlist = schema["properties"]["shortlist_labels"]
    assert shortlist["minItems"] == 4
    assert shortlist["maxItems"] == 12
    assert shortlist["items"]["enum"] == labels


def test_router_quality_keeps_clean_pretty_json_unpenalized():
    parsed = {
        "brief_analysis": "简要分析",
        "shortlist_labels": [f"标签{i}" for i in range(8)],
    }
    raw = json.dumps(parsed, ensure_ascii=False, indent=2)
    quality = router_output_quality(raw, parsed, parsed["shortlist_labels"], 8)
    assert quality["penalty"] == 0.0
    assert quality["abnormal_whitespace"] == 0.0
    assert quality["duplicate_excess"] == 0.0


def test_router_quality_softly_penalizes_duplicates_missing_and_whitespace():
    parsed = {
        "brief_analysis": "简要分析",
        "shortlist_labels": ["标签1", "标签1", "标签2", "标签2"],
    }
    raw = '{"brief_analysis":"简要分析","shortlist_labels":["标签1",\n\n\n\n"标签1","标签2","标签2"]}'
    quality = router_output_quality(raw, parsed, parsed["shortlist_labels"], 8)
    assert quality["duplicate_penalty"] == 0.02
    assert quality["missing_penalty"] == 0.02
    assert quality["whitespace_penalty"] == 0.01
    assert quality["penalty"] == 0.05
    assert quality["duplicate_excess"] == 2.0
    assert quality["missing_unique_count"] == 6.0
    assert quality["abnormal_whitespace"] == 1.0


def test_planner_quality_penalizes_cross_bucket_duplicates_and_whitespace():
    parsed = {
        "brief_analysis": "分析",
        "filtered_labels": {
            "tool_required_labels": [
                {"label": "导流", "reason": "需要工具", "required_tools": ["get_comments"]}
            ],
            "possible_labels": [
                {"label": "导流", "reason": "重复"},
                {"label": "通过", "reason": "可能通过"},
            ],
        },
    }
    raw = json.dumps(parsed, ensure_ascii=False).replace(",", ",\n\n\n\n", 1)
    quality = planner_output_quality(raw, parsed)
    assert quality["duplicate_excess"] == 1.0
    assert quality["cross_bucket_duplicate_count"] == 1.0
    assert quality["duplicate_penalty"] == 0.01
    assert quality["whitespace_penalty"] == 0.01
    assert quality["penalty"] == 0.02


def test_final_quality_penalizes_only_abnormal_layout_for_valid_single_label():
    parsed = {"predict_label": ["通过"], "audit_trace": "未命中风险"}
    clean = final_output_quality(json.dumps(parsed, ensure_ascii=False, indent=2), parsed)
    noisy = final_output_quality(
        '{"predict_label":["通过"],\n\n\n\n"audit_trace":"未命中风险"}',
        parsed,
    )
    assert clean["penalty"] == 0.0
    assert noisy["duplicate_excess"] == 0.0
    assert noisy["whitespace_penalty"] == 0.01
    assert noisy["penalty"] == 0.01


def test_planner_schema_limits_labels_and_tools():
    response_format = planner_response_format(["通过", "导流"], ["get_comments"])
    schema = response_format["json_schema"]["schema"]
    filtered = schema["properties"]["filtered_labels"]
    assert list(filtered["properties"]) == ["tool_required_labels", "possible_labels"]
    possible = filtered["properties"]["possible_labels"]
    assert possible["items"]["properties"]["label"]["enum"] == ["通过", "导流"]
    required_tools = filtered["properties"]["tool_required_labels"]["items"]["properties"]["required_tools"]
    assert required_tools["minItems"] == 1
    assert required_tools["items"]["enum"] == ["get_comments"]
    assert "tool_calls" not in schema["properties"]


def test_final_candidates_keep_pass_and_fallback_to_shortlist():
    labels = build_final_candidate_labels(
        ["通过", "导流", "商业推广"],
        ["通过", "导流"],
        [{"label": "导流"}],
        [],
    )
    assert labels == ["导流", "通过"]
    final_schema = final_response_format(labels)["json_schema"]["schema"]
    assert final_schema["properties"]["predict_label"]["items"]["enum"] == labels


def test_planner_derives_deduplicated_calls_from_required_tools():
    client = MockLLMClient()
    client.register(
        "planner",
        json.dumps(
            {
                "brief_analysis": "需要工具确认导流",
                "filtered_labels": {
                    "tool_required_labels": [
                        {
                            "label": "导流",
                            "reason": "需要查看评论和历史",
                            "required_tools": ["get_comments", "get_recent_notes"],
                        }
                    ],
                    "possible_labels": [
                        {"label": "导流", "reason": "重复项应被移除"},
                        {"label": "通过", "reason": "仍可能通过"},
                    ],
                },
            },
            ensure_ascii=False,
        ),
    )
    planner = PlannerAgent(client, default_loader())
    output, _ = planner.run_plan(
        AuditInput(note="帖子", candidate_labels=["导流", "通过"]),
        shortlist_labels=["导流", "通过"],
        shortlist_rules=[],
        available_tools=[
            {"name": "get_comments", "brief": "评论", "when": "检查评论"},
            {"name": "get_recent_notes", "brief": "近期笔记", "when": "检查历史"},
        ],
    )
    assert [entry.label for entry in output.tool_required_labels] == ["导流"]
    assert [entry.label for entry in output.possible_labels] == ["通过"]
    assert [call.tool_name for call in output.tool_calls] == [
        "get_comments",
        "get_recent_notes",
    ]
