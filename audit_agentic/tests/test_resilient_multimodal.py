from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from audit_agentic.agents.multimodal import (
    IMAGE_TAG,
    assert_image_alignment,
    prepare_accessible_multimodal_inputs,
)


def _jpeg(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    Image.new("RGB", size, (20, 80, 140)).save(path, format="JPEG")
    return path


def test_bad_image_removes_only_its_matching_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_RESIZED_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    first = _jpeg(tmp_path / "first.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    third = _jpeg(tmp_path / "third.jpg")

    prepared = prepare_accessible_multimodal_inputs(
        f"A{IMAGE_TAG}B{IMAGE_TAG}C{IMAGE_TAG}D",
        [str(first), str(bad), str(third)],
        image_max_tokens=448,
    )

    assert prepared.text == f"A{IMAGE_TAG}BC{IMAGE_TAG}D"
    assert prepared.images == [str(first), str(third)]
    assert prepared.dropped_images == [str(bad)]
    assert_image_alignment(prepared.text, prepared.images)


def test_missing_images_and_extra_images_are_dropped_positionally(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_RESIZED_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    image = _jpeg(tmp_path / "only.jpg")
    extra = _jpeg(tmp_path / "extra.jpg")

    prepared = prepare_accessible_multimodal_inputs(
        f"left{IMAGE_TAG}middle{IMAGE_TAG}right",
        [str(image), str(extra), str(extra)],
        image_max_tokens=448,
    )

    assert prepared.text == f"left{IMAGE_TAG}middle{IMAGE_TAG}right"
    assert prepared.images == [str(image), str(extra)]
    assert prepared.dropped_images == [str(extra)]


def test_relax_mode_keeps_valid_local_native_image_on_original_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_RESIZED_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIT_DISABLE_LOCAL_IMAGE_RESIZE", "1")
    image = _jpeg(tmp_path / "large.jpg", size=(2048, 2048))

    prepared = prepare_accessible_multimodal_inputs(
        f"before{IMAGE_TAG}after",
        [str(image)],
        image_max_tokens=64,
    )

    assert prepared.images == [str(image)]
    assert prepared.converted_images == []
    assert not list((tmp_path / "cache" / "online_prepared").rglob("*.jpg"))


def test_ffmpeg_fallback_converts_heif_like_input(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_RESIZED_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "source.heif"
    source.write_bytes(b"fake heif payload")

    def fake_run(command, **kwargs):
        _jpeg(Path(command[-1]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("audit_agentic.agents.multimodal.subprocess.run", fake_run)
    prepared = prepare_accessible_multimodal_inputs(
        f"before{IMAGE_TAG}after",
        [str(source)],
        image_max_tokens=448,
    )

    assert prepared.text == f"before{IMAGE_TAG}after"
    assert len(prepared.images) == 1
    assert prepared.images[0].endswith(".jpg")
    assert Path(prepared.images[0]).exists()
    assert prepared.converted_images == [str(source)]
