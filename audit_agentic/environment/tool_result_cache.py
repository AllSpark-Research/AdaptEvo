"""File-backed cache for audit tool results.

The agentic rollout launches one Python process per managed session. In RL
training the same note can be sampled multiple times, so tool calls are highly
repeated. This cache stores raw tool ``run()`` results by tool name + merged
args + selected data-env vars. Rendering still happens after cache lookup, so
``<image>`` / ``display_images`` alignment remains governed by the current
renderer code.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


_CACHE_ENV_KEYS = (
    "BLOB_INDEX_PATH",
    "IMAGE_CACHE_DIR",
    "PLAGIARISM_CACHE_PATH",
    "AUDIT_TOOL_DATA_PATH",
    "RECENT_NOTES_MAX_NOTES",
    "RECENT_NOTES_MAX_IMAGES_PER_NOTE",
    "COMMERCIAL_DETAIL_MAX_IMAGES_PER_ITEM",
)


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "cache" / "tool_results"


def cache_disabled() -> bool:
    from .data_access_mode import online_data_access_enabled

    if online_data_access_enabled():
        return True
    return os.environ.get("AUDIT_TOOL_RESULT_CACHE_DISABLE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def cache_dir() -> Path:
    return Path(os.environ.get("AUDIT_TOOL_RESULT_CACHE_PATH") or _default_cache_dir())


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


def _identity_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """丢掉 `_` 前缀的运行时参数，只保留决定结果的语义参数。

    这些参数由调用方注入、不影响工具结果，但曾经进过 cache key：
      * `_audit_context` —— 只有 AgentScope 路线会绑（agentscope_audit_agent.py），
        legacy agent.py 和 prewarm 脚本都不绑。留在 key 里会让两条路各算各的
        哈希，AgentScope 100% miss 掉预热好的缓存。
      * `_calls_blob`    —— 由 note_id 唯一决定，冗余且体积巨大。
      * `_image_cache_dir` —— 已由 `IMAGE_CACHE_DIR` 这个 env key 覆盖。

    legacy / prewarm 本来就不传下划线参数，所以它们的 key 不变，现存缓存不失效。
    """
    identity = {
        key: value
        for key, value in (args or {}).items()
        if not str(key).startswith("_")
    }
    mode = os.environ.get("AUDIT_TOOL_CACHE_IDENTITY_MODE", "default").strip().lower()
    if mode == "note_history":
        # The human-cot-0818-v1 cache is shared across audit sources and keyed
        # exclusively by the immutable note/history pair.
        return {
            key: identity[key]
            for key in ("note_id", "history_id")
            if key in identity
        }
    return identity


def cache_key_payload(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    mode = os.environ.get("AUDIT_TOOL_CACHE_IDENTITY_MODE", "default").strip().lower()
    cache_env = {
        key: ("" if mode == "note_history" else os.environ.get(key, ""))
        for key in _CACHE_ENV_KEYS
    }
    return {
        "version": os.environ.get("AUDIT_TOOL_CACHE_VERSION", "tool-cache-v1"),
        "tool_name": str(tool_name),
        "args": _json_safe(_identity_args(args)),
        "env": cache_env,
    }


def cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    payload = cache_key_payload(tool_name, args)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_file(tool_name: str, key: str) -> Path:
    safe_tool = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(tool_name))
    return cache_dir() / safe_tool / key[:2] / key[2:4] / f"{key}.json"


def get_cached_result(tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if cache_disabled():
        return None
    key = cache_key(tool_name, args)
    path = cache_file(tool_name, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    result = payload.get("result") if isinstance(payload, dict) else None
    return result if isinstance(result, dict) else None


def set_cached_result(tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    if cache_disabled() or not isinstance(result, dict):
        return
    key = cache_key(tool_name, args)
    path = cache_file(tool_name, key)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "key": key,
        "key_payload": cache_key_payload(tool_name, args),
        "result": _json_safe(result),
    }
    tmp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
