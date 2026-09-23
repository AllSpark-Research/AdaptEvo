from __future__ import annotations

from pathlib import Path

import pytest

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import URLSource

from audit_agentic.agents import multimodal


def test_bad_image_drops_only_matching_placeholder(monkeypatch) -> None:
    def prepare(
        image_ref: str,
        image_max_tokens: int,
        *,
        force_local_resize: bool = False,
    ):
        del image_max_tokens, force_local_resize
        if image_ref == "bad-image":
            return None, False
        return "/cache/good.jpg", True

    monkeypatch.setattr(multimodal, "_prepare_accessible_image", prepare)
    prepared = multimodal.prepare_accessible_multimodal_inputs(
        "<image>first\n<image>second",
        ["bad-image", "good-image"],
        image_max_tokens=448,
    )

    assert prepared.text == "first\n<image>second"
    assert prepared.images == ["/cache/good.jpg"]
    assert prepared.dropped_images == ["bad-image"]
    assert prepared.converted_images == ["good-image"]
    multimodal.assert_image_alignment(prepared.text, prepared.images)


def test_agentscope_formatter_skips_missing_local_image(tmp_path) -> None:
    formatter = OpenAIChatFormatter()
    source = URLSource(
        url=f"file://{tmp_path / 'missing.jpg'}",
        media_type="image/jpeg",
    )

    assert formatter._format_image_source(source) is None


def test_ray_runtime_env_includes_bad_image_and_protocol_controls() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "train"
        / "audit_agentic_base.sh"
    )
    if not path.exists():
        pytest.skip("Internal scheduler launcher is excluded from the public snapshot")
    script = path.read_text(encoding="utf-8")
    expected_names = {
        "AUDIT_SKIP_UNAVAILABLE_IMAGES",
        "AUDIT_BAD_IMAGE_CACHE_TTL_SECONDS",
        "AUDIT_IMAGE_DOWNLOAD_TIMEOUT",
        "AUDIT_IMAGE_DOWNLOAD_MAX_BYTES",
        "AUDIT_RESIZED_IMAGE_CACHE_DIR",
        "AUDIT_DROP_REMOTE_IMAGES",
        "AUDIT_AGENTSCOPE_TOOL_PROTOCOL_PENALTY",
        "AUDIT_AGENTSCOPE_TOOL_PROTOCOL_PENALTY_CAP",
        "AUDIT_AGENTSCOPE_INVALID_FINAL_REWARD",
        "AUDIT_AGENTSCOPE_REWARD_FLOOR",
    }
    assert all(f'"{name}"' in script for name in expected_names)
