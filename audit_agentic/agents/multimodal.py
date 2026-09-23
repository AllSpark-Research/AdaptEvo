"""Utility for building multimodal (text + image) messages.

Splits a text prompt containing ``<image>`` markers into an OpenAI-compatible
multimodal content list, interleaving text and image_url parts.

``image_max_tokens`` can be applied before sending the request by resizing local
images into a cache directory. For Relax training, this local resize path should
be disabled so rollout uses the original image paths and relies on Relax's
``--image-max-token-num`` preprocessing.
"""

from __future__ import annotations

import base64
import hashlib
import math
import mimetypes
import os
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

IMAGE_TAG = "<image>"
DEFAULT_IMAGE_MAX_TOKENS = 512
QWEN_VL_PATCH_FACTOR = 32
DEFAULT_RESIZED_IMAGE_CACHE_DIR = str(
    Path(__file__).resolve().parents[1] / "environment" / "cache" / "resized_images"
)
_PIL_OPEN_LOCK = threading.Lock()
_IMAGE_PREP_LOCKS_GUARD = threading.Lock()
_IMAGE_PREP_LOCKS: Dict[str, threading.Lock] = {}
_MODEL_NATIVE_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}
_PREPARED_IMAGE_CACHE_VERSION = "v1-online-safe-image"


@dataclass
class PreparedMultimodalInputs:
    """Aligned multimodal inputs after removing unusable images."""

    text: str
    images: List[str]
    dropped_images: List[str]
    converted_images: List[str]


class ImageAlignmentError(ValueError):
    """Raised when prompt <image> placeholders and image list are misaligned."""


def count_image_tags(text: str) -> int:
    return str(text or "").count(IMAGE_TAG)


def assert_image_alignment(text: str, images: Optional[List[str]], context: str = "multimodal message") -> None:
    """Fail fast if a prompt's <image> placeholders do not match image inputs."""
    n_tags = count_image_tags(text)
    n_images = len(images or [])
    if n_tags == n_images:
        return
    preview = str(text or "").replace("\n", "\\n")[:240]
    raise ImageAlignmentError(
        f"{context}: <image> placeholders ({n_tags}) != images ({n_images}); "
        f"prompt preview={preview!r}"
    )


def _cache_dir() -> Path:
    return Path(os.environ.get("AUDIT_RESIZED_IMAGE_CACHE_DIR", DEFAULT_RESIZED_IMAGE_CACHE_DIR))


def resilient_image_preparation_enabled() -> bool:
    """Whether unavailable images should be removed before model requests."""
    from ..environment.data_access_mode import online_data_access_enabled

    if online_data_access_enabled():
        return True
    return os.environ.get("AUDIT_SKIP_UNAVAILABLE_IMAGES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _negative_cache_active(path: Path) -> bool:
    if not path.exists():
        return False
    ttl = _float_env("AUDIT_BAD_IMAGE_CACHE_TTL_SECONDS", 600.0)
    if ttl <= 0:
        return False
    try:
        return time.time() - path.stat().st_mtime <= ttl
    except OSError:
        return False


def _touch_marker(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _wait_for_nonempty_file(path: Path) -> bool:
    """Wait briefly for a newly published shared-cache file to be visible."""
    timeout = _float_env("AUDIT_IMAGE_CACHE_VISIBILITY_TIMEOUT", 2.0, 0.0)
    deadline = time.monotonic() + timeout
    while True:
        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _model_native_image_readable(path: str) -> bool:
    """Validate a local image without decoding its full pixel payload."""
    if Path(path).suffix.lower() in {".heic", ".heif"}:
        return False
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            image = _open_image_for_guarded_resize(path, Image)
        with image:
            image_format = str(image.format or "").upper()
            image.verify()
        return image_format in _MODEL_NATIVE_IMAGE_FORMATS
    except Exception:
        return False


def _image_preparation_lock(image_ref: str) -> threading.Lock:
    key = hashlib.sha256(str(image_ref).encode("utf-8")).hexdigest()
    with _IMAGE_PREP_LOCKS_GUARD:
        return _IMAGE_PREP_LOCKS.setdefault(key, threading.Lock())


def _strict_remote_cache_paths(url: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    root = _cache_dir() / "online_remote" / digest[:2]
    return root / f"{digest}.image", root / f"{digest}.bad"


def _download_remote_image_strict(image_ref: str) -> Optional[str]:
    """Download once without a separate HEAD request and cap response bytes."""
    if not image_ref.startswith(("http://", "https://")):
        return None

    target, bad_marker = _strict_remote_cache_paths(image_ref)
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    if _negative_cache_active(bad_marker):
        return None

    try:
        import requests
    except Exception:
        _touch_marker(bad_marker)
        return None

    tmp_path: Optional[Path] = None
    max_bytes = _int_env("AUDIT_REMOTE_IMAGE_MAX_BYTES", 32 * 1024 * 1024)
    connect_timeout = _float_env("AUDIT_REMOTE_IMAGE_CONNECT_TIMEOUT", 3.0, 0.1)
    read_timeout = _float_env("AUDIT_REMOTE_IMAGE_READ_TIMEOUT", 10.0, 0.1)
    response = None
    try:
        response = requests.get(
            image_ref,
            stream=True,
            timeout=(connect_timeout, read_timeout),
            headers={"User-Agent": "audit-agentic-online-image/1.0"},
        )
        if response.status_code != 200:
            _touch_marker(bad_marker)
            return None
        try:
            content_length = int(response.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            content_length = 0
        if max_bytes and content_length > max_bytes:
            _touch_marker(bad_marker)
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
        size = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if max_bytes and size > max_bytes:
                    _touch_marker(bad_marker)
                    return None
                handle.write(chunk)
        if size <= 0:
            _touch_marker(bad_marker)
            return None
        os.replace(tmp_path, target)
        return str(target)
    except Exception:
        _touch_marker(bad_marker)
        return None
    finally:
        try:
            if response is not None:
                response.close()
        except Exception:
            pass
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _prepared_image_paths(path: str, stat: os.stat_result, image_max_tokens: int) -> tuple[Path, Path, Path]:
    key = (
        f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{image_max_tokens}|"
        f"{_PREPARED_IMAGE_CACHE_VERSION}"
    ).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    root = _cache_dir() / "online_prepared" / digest[:2]
    return (
        root / f"{digest}.jpg",
        root / f"{digest}.ok",
        root / f"{digest}.bad",
    )


def _save_pillow_as_jpeg(
    path: str,
    output_path: Path,
    image_max_tokens: int,
    force_jpeg: bool = False,
) -> Optional[str]:
    """Validate an image and normalize/resize it only when necessary."""
    try:
        from PIL import Image, ImageOps
    except Exception:
        return None

    max_pixels = max(1, int(image_max_tokens)) * QWEN_VL_PATCH_FACTOR * QWEN_VL_PATCH_FACTOR
    tmp_path: Optional[Path] = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            img_ctx = _open_image_for_guarded_resize(path, Image)
        with img_ctx as img:
            image_format = str(img.format or "").upper()
            width, height = img.size
            orientation = _read_exif_orientation(img)
            swaps_axes = orientation in {5, 6, 7, 8}
            display_w, display_h = (height, width) if swaps_axes else (width, height)
            new_w, new_h = _resize_dimensions(display_w, display_h, max_pixels)
            suffix = Path(path).suffix.lower()
            requires_jpeg = (
                force_jpeg
                or image_format not in _MODEL_NATIVE_IMAGE_FORMATS
                or suffix in {".heic", ".heif"}
                or (new_w, new_h) != (display_w, display_h)
            )
            if not requires_jpeg:
                img.load()
                return path

            if image_format in {"JPEG", "MPO"}:
                draft_size = (new_h, new_w) if swaps_axes else (new_w, new_h)
                img.draft("RGB", draft_size)
            img.load()
            img = _safe_exif_transpose(img, ImageOps)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert("RGB")
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
            if img.size != (new_w, new_h):
                img = img.resize((new_w, new_h), resampling)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_name(
                f"{output_path.stem}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            img.save(tmp_path, format="JPEG", quality=90, optimize=True)
            os.replace(tmp_path, output_path)
            return str(output_path)
    except Exception:
        return None
    finally:
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _convert_with_ffmpeg(path: str, output_path: Path) -> Optional[str]:
    """Fallback decoder for HEIF/HEIC images when Pillow lacks a plugin."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    intermediate = output_path.with_name(
        f"{output_path.stem}.{os.getpid()}.{uuid4().hex}.ffmpeg.jpg"
    )
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                path,
                "-frames:v",
                "1",
                str(intermediate),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_float_env("AUDIT_IMAGE_CONVERT_TIMEOUT", 20.0, 1.0),
        )
        if completed.returncode != 0 or not intermediate.exists() or intermediate.stat().st_size <= 0:
            return None
        return str(intermediate)
    except (OSError, subprocess.SubprocessError):
        return None


def _prepare_accessible_image(
    image_ref: str,
    image_max_tokens: int,
    *,
    force_local_resize: bool = False,
) -> tuple[Optional[str], bool]:
    """Return a validated model-compatible image path and conversion status."""
    if not image_ref:
        return None, False
    with _image_preparation_lock(image_ref):
        is_remote = image_ref.startswith(("http://", "https://"))
        path = _download_remote_image_strict(image_ref) if is_remote else _local_path_from_image_ref(image_ref)
        if not path:
            return None, False

        try:
            stat = os.stat(path)
        except OSError:
            return None, False

        output_path, ok_marker, bad_marker = _prepared_image_paths(path, stat, image_max_tokens)
        # Relax/SGLang applies IMAGE_MAX_TOKENS in its own multimodal processor.
        # Keep valid local native images on their original path so concurrent
        # rollout processes do not all publish the same derived JPEG on JuiceFS.
        if not is_remote and _local_resize_disabled() and not force_local_resize:
            if ok_marker.exists():
                return path, False
            if _negative_cache_active(bad_marker):
                return None, False
            if _model_native_image_readable(path):
                _touch_marker(ok_marker)
                return path, False

        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path), True
        if ok_marker.exists():
            return path, False
        if _negative_cache_active(bad_marker):
            return None, False

        # Remote files are normalized to JPEG even when already decodable.
        # This avoids MIME/extension mismatches from service URLs and leaves a
        # reusable model-ready artifact for later rollout samples.
        prepared = _save_pillow_as_jpeg(
            path,
            output_path,
            image_max_tokens,
            force_jpeg=is_remote,
        )
        if prepared:
            if prepared == path:
                _touch_marker(ok_marker)
                return path, False
            if _wait_for_nonempty_file(Path(prepared)):
                return prepared, True
            if not is_remote and _model_native_image_readable(path):
                _touch_marker(ok_marker)
                return path, False
            return None, False

        ffmpeg_image = _convert_with_ffmpeg(path, output_path)
        if ffmpeg_image:
            try:
                prepared = _save_pillow_as_jpeg(
                    ffmpeg_image,
                    output_path,
                    image_max_tokens,
                    force_jpeg=True,
                )
                if prepared and _wait_for_nonempty_file(Path(prepared)):
                    return prepared, True
            finally:
                try:
                    Path(ffmpeg_image).unlink()
                except OSError:
                    pass

        _touch_marker(bad_marker)
        return None, False


def prepare_accessible_multimodal_inputs(
    text: str,
    images: List[str],
    image_max_tokens: int = 448,
    *,
    force_local_resize: bool = False,
) -> PreparedMultimodalInputs:
    """Drop bad images and only their positionally corresponding placeholders.

    This function also removes unmatched placeholders and ignores extra image
    references.  Its output is always safe to pass to the strict alignment
    check in ``build_multimodal_content``.
    """
    text = str(text or "")
    images = [str(value) for value in (images or [])]
    parts = text.split(IMAGE_TAG)
    placeholder_count = len(parts) - 1
    rebuilt: List[str] = []
    kept: List[str] = []
    dropped: List[str] = []
    converted: List[str] = []

    for index in range(placeholder_count):
        rebuilt.append(parts[index])
        if index >= len(images):
            continue
        image_ref = images[index]
        prepared, was_converted = _prepare_accessible_image(
            image_ref,
            image_max_tokens,
            force_local_resize=force_local_resize,
        )
        if not prepared:
            dropped.append(image_ref)
            continue
        rebuilt.append(IMAGE_TAG)
        kept.append(prepared)
        if was_converted:
            converted.append(image_ref)
    rebuilt.append(parts[-1])
    if len(images) > placeholder_count:
        dropped.extend(images[placeholder_count:])

    aligned_text = "".join(rebuilt)
    assert_image_alignment(aligned_text, kept, "prepare_accessible_multimodal_inputs")
    return PreparedMultimodalInputs(
        text=aligned_text,
        images=kept,
        dropped_images=dropped,
        converted_images=converted,
    )


def _local_resize_disabled() -> bool:
    return os.environ.get("AUDIT_DISABLE_LOCAL_IMAGE_RESIZE", "").strip().lower() in {"1", "true", "yes", "on"}


def _drop_remote_images() -> bool:
    return os.environ.get("AUDIT_DROP_REMOTE_IMAGES", "").strip().lower() in {"1", "true", "yes", "on"}


def _images_as_base64() -> bool:
    """When true, local images are inlined into image_url as data URLs.

    Enable when the LLM endpoint is a cloud API that cannot fetch file:// URIs
    (e.g. doubao MaaS). Set AUDIT_IMAGES_AS_BASE64=1 to opt in.
    """
    return os.environ.get("AUDIT_IMAGES_AS_BASE64", "").strip().lower() in {"1", "true", "yes", "on"}


def _images_base64_max_bytes() -> int:
    """Hard cap on per-image bytes before base64 encoding. Falls back to
    file:// when exceeded so we never explode the request body.
    """
    try:
        return max(0, int(os.environ.get("AUDIT_IMAGES_BASE64_MAX_BYTES", str(5 * 1024 * 1024))))
    except ValueError:
        return 5 * 1024 * 1024


def _try_encode_data_url(path: str) -> Optional[str]:
    """Read a local image and return an OpenAI-compatible data URL.

    Returns None on any I/O / size-cap failure so caller can fall back.
    """
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    cap = _images_base64_max_bytes()
    if cap and stat.st_size > cap:
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    mime = mimetypes.guess_type(path)[0]
    # audit_agentic writes cached files without ``.jpg`` half the time; default
    # to jpeg since ``_resize_image_for_budget`` always saves JPEG.
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _local_path_from_image_ref(image_ref: str) -> Optional[str]:
    if not image_ref:
        return None
    if image_ref.startswith(("http://", "https://")):
        return None
    if image_ref.startswith("file://"):
        return image_ref[len("file://") :]
    return image_ref


def _extension_for_url(url: str, content_type: str = "") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def _remote_cache_path(url: str, content_type: str = "") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return _cache_dir() / "remote" / digest[:2] / f"{digest}{_extension_for_url(url, content_type)}"


def _download_remote_image(image_ref: str) -> Optional[str]:
    if not image_ref.startswith(("http://", "https://")):
        return None

    cached = _remote_cache_path(image_ref)
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)

    try:
        import requests
    except Exception:
        return None

    try:
        resp = requests.get(
            image_ref,
            stream=True,
            timeout=float(os.environ.get("AUDIT_REMOTE_IMAGE_DOWNLOAD_TIMEOUT", "15")),
            headers={"User-Agent": "audit-agentic-image-resizer/1.0"},
        )
        if resp.status_code != 200:
            return None
        target = _remote_cache_path(image_ref, resp.headers.get("Content-Type", ""))
        if target.exists() and target.stat().st_size > 0:
            return str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.stem}.{os.getpid()}.{uuid4().hex}.tmp")
        size = 0
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                size += len(chunk)
                f.write(chunk)
        if size <= 0:
            tmp_path.unlink(missing_ok=True)
            return None
        os.replace(tmp_path, target)
        return str(target)
    except Exception:
        return None
    finally:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _resize_dimensions(width: int, height: int, max_pixels: int, factor: int = QWEN_VL_PATCH_FACTOR) -> tuple[int, int]:
    if width <= 0 or height <= 0 or max_pixels <= 0 or width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / float(width * height))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    if factor > 1:
        new_w = max(factor, (new_w // factor) * factor)
        new_h = max(factor, (new_h // factor) * factor)

    while new_w * new_h > max_pixels and (new_w > factor or new_h > factor):
        if new_w >= new_h and new_w > factor:
            new_w -= factor
        elif new_h > factor:
            new_h -= factor
        else:
            break
    return max(1, new_w), max(1, new_h)


def _resized_cache_path(path: str, stat: os.stat_result, max_pixels: int) -> Path:
    key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{max_pixels}|v2-safe-decode".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return _cache_dir() / digest[:2] / f"{digest}.jpg"


def _read_exif_orientation(img: Any) -> int:
    """Read EXIF orientation without letting malformed metadata abort resize."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            return int((img.getexif() or {}).get(274, 1))
    except Exception:
        return 1


def _safe_exif_transpose(img: Any, image_ops: Any) -> Any:
    """Apply valid EXIF orientation and strip malformed metadata afterwards."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            img = image_ops.exif_transpose(img)
    except Exception:
        pass
    try:
        img.info.pop("exif", None)
    except Exception:
        pass
    return img


def _open_image_for_guarded_resize(path: str, image_module: Any) -> Any:
    """Open lazily so our own pixel-budget logic can run before decode.

    Pillow's global bomb check can raise before callers can inspect dimensions.
    Keep the global override inside a short lock and restore it immediately;
    no pixel data is decoded until after format/size checks and JPEG draft().
    """
    with _PIL_OPEN_LOCK:
        old_limit = image_module.MAX_IMAGE_PIXELS
        image_module.MAX_IMAGE_PIXELS = None
        try:
            return image_module.open(path)
        finally:
            image_module.MAX_IMAGE_PIXELS = old_limit


def _resize_image_for_budget(image_ref: str, image_max_tokens: int) -> str:
    """Resize a local image for Qwen-VL visual-token budget and return a path.

    Remote URLs are first downloaded into the resize cache when local resizing
    is enabled. Failures return the original reference so preprocessing cannot
    break the audit flow.
    """
    if image_max_tokens <= 0 or _local_resize_disabled():
        return image_ref

    path = _download_remote_image(image_ref) if image_ref.startswith(("http://", "https://")) else None
    path = path or _local_path_from_image_ref(image_ref)
    if not path:
        return image_ref

    try:
        stat = os.stat(path)
    except OSError:
        return image_ref

    max_pixels = int(image_max_tokens) * QWEN_VL_PATCH_FACTOR * QWEN_VL_PATCH_FACTOR
    try:
        from PIL import Image, ImageOps
    except Exception:
        return image_ref

    out_path = _resized_cache_path(path, stat, max_pixels)
    if out_path.exists():
        return str(out_path)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            img_ctx = _open_image_for_guarded_resize(path, Image)
        with img_ctx as img:
            image_format = str(img.format or "").upper()
            width, height = img.size
            orientation = _read_exif_orientation(img)
            swaps_axes = orientation in {5, 6, 7, 8}
            display_w, display_h = (height, width) if swaps_axes else (width, height)
            new_w, new_h = _resize_dimensions(display_w, display_h, max_pixels)
            if new_w == display_w and new_h == display_h:
                return image_ref

            # JPEG/MPO can decode directly at 1/2, 1/4, or 1/8 resolution.
            # This preserves the complete source image while avoiding a full
            # hundreds-of-megapixels RGB allocation before the final resize.
            if image_format in {"JPEG", "MPO"}:
                draft_size = (new_h, new_w) if swaps_axes else (new_w, new_h)
                img.draft("RGB", draft_size)
            img.load()
            img = _safe_exif_transpose(img, ImageOps)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
            if img.size != (new_w, new_h):
                img = img.resize((new_w, new_h), resampling)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_name(f"{out_path.stem}.{os.getpid()}.{uuid4().hex}.tmp")
            img.save(tmp_path, format="JPEG", quality=92, optimize=True)
            os.replace(tmp_path, out_path)
            return str(out_path)
    except Exception:
        return image_ref
    finally:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _image_ref_to_url(image_ref: str) -> str:
    """Return an OpenAI image_url value, normalizing local refs to absolute file URIs."""
    if image_ref.startswith(("http://", "https://")):
        return image_ref
    if image_ref.startswith("file://"):
        path = image_ref[len("file://") :]
        if path and not Path(path).is_absolute():
            path = str(Path(path).resolve())
        if _images_as_base64():
            data_url = _try_encode_data_url(path)
            if data_url:
                return data_url
        return f"file://{path}" if path else image_ref
    resolved = str(Path(image_ref).resolve())
    if _images_as_base64():
        data_url = _try_encode_data_url(resolved)
        if data_url:
            return data_url
    return f"file://{resolved}"


def build_multimodal_content(
    text: str,
    images: List[str],
    image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    """Split text on ``<image>`` markers and interleave with image_url parts.

    If ``images`` is empty or there are no ``<image>`` tags, returns a plain
    text content list (single element).

    Each local image path is resized if needed, then converted to a ``file://``
    URL. The inference backend (sglang/vLLM) reads the file directly from the
    shared filesystem.
    """
    images = list(images or [])
    assert_image_alignment(text, images, "build_multimodal_content")
    if not images:
        return [{"type": "text", "text": text}]

    parts = text.split(IMAGE_TAG)
    content: List[Dict[str, Any]] = []
    img_idx = 0

    for i, part in enumerate(parts):
        # Only emit a text part if it has non-whitespace characters. Relax's
        # check_messages rejects content items whose text strips to empty
        # (raises HTTP 400 → session never produces chat IR). When two
        # <image> tags are separated only by "\n", we still skip the empty
        # text part — the model sees two consecutive image parts, which is
        # the canonical OpenAI multimodal format.
        if part and part.strip():
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1 and img_idx < len(images):
            img_path = _resize_image_for_budget(images[img_idx], image_max_tokens)
            if _drop_remote_images() and img_path.startswith(("http://", "https://")):
                img_idx += 1
                continue
            img_url = _image_ref_to_url(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_url,
                },
            })
            img_idx += 1

    return content


def make_multimodal_message(
    role: str,
    text: str,
    images: Optional[List[str]] = None,
    image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
) -> Dict[str, Any]:
    """Build a single message dict, multimodal if images are present."""
    images = list(images or [])
    assert_image_alignment(text, images, f"{role} message")
    if not images:
        return {"role": role, "content": text}
    content = build_multimodal_content(text, images, image_max_tokens)
    return {"role": role, "content": content}



def collect_rule_images(rules: Any) -> List[str]:
    """Collect RuleInfo.rule_images in prompt order from objects or dicts."""
    images: List[str] = []
    for rule in rules or []:
        if isinstance(rule, dict):
            values = rule.get("rule_images") or []
        else:
            values = getattr(rule, "rule_images", []) or []
        if isinstance(values, list):
            images.extend(str(v) for v in values if v)
    return images
