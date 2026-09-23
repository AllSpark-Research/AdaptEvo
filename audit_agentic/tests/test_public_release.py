import pytest

from audit_agentic.environment.rule_retriever import (
    _append_source_specific_queue_notice,
    default_mock_rules,
)
from audit_agentic.environment.services.get_new_url import try_transfer_blocked_url
from audit_agentic.environment.services.query_factor import query_factor


def test_no_embedded_source_specific_business_rule():
    assert _append_source_specific_queue_notice("example_queue_004", "demo") == "demo"
    assert set(default_mock_rules()) == {"synthetic"}


def test_media_resolver_is_opt_in(monkeypatch):
    monkeypatch.delenv("AUDIT_MEDIA_RESOLVER_URL", raising=False)
    assert try_transfer_blocked_url("https://example.invalid/image.png") == "https://example.invalid/image.png"


def test_factor_service_requires_deployment_endpoint(monkeypatch):
    monkeypatch.delenv("AUDIT_FACTOR_SERVICE_URL", raising=False)
    with pytest.raises(RuntimeError, match="AUDIT_FACTOR_SERVICE_URL"):
        query_factor("synthetic-factor", {})


def test_synthetic_tool_rendering():
    from examples.synthetic_audit import LookupSyntheticEvidence

    tool = LookupSyntheticEvidence()
    result = tool.run({})
    assert result["verified"] is True
    rendered = tool.render(result)
    assert rendered.images == []
    assert "fictional example" in rendered.text
