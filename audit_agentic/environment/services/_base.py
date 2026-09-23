"""DataBackend: 统一的服务数据获取接口（在线调 Mars / 离线从 calls_blob 解析）。"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ONLINE_DISABLED_WARNED = False
_ONLINE_SESSION_LOCAL = threading.local()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _online_session(pool_size: int):
    """Reuse one requests session per worker thread for keep-alive/TLS reuse."""
    import requests
    from requests.adapters import HTTPAdapter

    sessions = getattr(_ONLINE_SESSION_LOCAL, "sessions", None)
    if sessions is None:
        sessions = {}
        _ONLINE_SESSION_LOCAL.sessions = sessions
    session = sessions.get(pool_size)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0,
            pool_block=False,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "contact@example.invalid",
            },
        )
        sessions[pool_size] = session
    return session


def online_services_disabled() -> bool:
    """Disable business-service access unless explicitly enabled by deployment."""
    from ..data_access_mode import online_data_access_enabled

    if online_data_access_enabled():
        return False
    return os.environ.get("AUDIT_DISABLE_ONLINE_SERVICES", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def warn_online_disabled_once(what: str) -> None:
    """整个进程只告警一次，避免每个工具调用都刷屏。"""
    global _ONLINE_DISABLED_WARNED
    if not _ONLINE_DISABLED_WARNED:
        _ONLINE_DISABLED_WARNED = True
        logger.warning(
            "AUDIT_DISABLE_ONLINE_SERVICES is on; skipping every online service "
            "call (first skipped: %s). Results degrade to offline data only.",
            what,
        )


def _maybe_json(v: Any) -> Any:
    if isinstance(v, str) and len(v) > 1 and v[0] in ("{", "["):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
    return v


class DataBackend(ABC):
    @abstractmethod
    def call(self, service_name: str, **params) -> Optional[Dict[str, Any]]:
        """调用一个服务，返回解析后的 response dict。失败返回 None。"""


class OnlineBackend(DataBackend):
    """在线模式：调 Mars HTTP API。"""

    def __init__(
        self,
        timeout: float | None = None,
        max_attempts: int | None = None,
        retry_wait: float | None = None,
    ):
        from tenacity import retry, stop_after_attempt, wait_fixed

        self._base = os.environ.get(
            "AUDIT_ONLINE_SERVICE_BASE",
            "",
        ).rstrip("/")
        self._timeout = max(
            0.1,
            float(
                timeout
                if timeout is not None
                else _env_float("AUDIT_ONLINE_SERVICE_TIMEOUT", 20.0)
            ),
        )
        attempts = max(
            1,
            int(
                max_attempts
                if max_attempts is not None
                else _env_int("AUDIT_ONLINE_SERVICE_MAX_ATTEMPTS", 3)
            ),
        )
        wait_seconds = max(
            0.0,
            float(
                retry_wait
                if retry_wait is not None
                else _env_float("AUDIT_ONLINE_SERVICE_RETRY_WAIT", 1.0)
            ),
        )
        pool_size = max(1, _env_int("AUDIT_ONLINE_HTTP_POOL_SIZE", 32))
        self._session = _online_session(pool_size)

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_fixed(wait_seconds),
            reraise=True,
        )
        def _post(url, body):
            resp = self._session.post(
                url,
                json=body,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()

        self._post = _post

    def call(self, service_name: str, **params) -> Optional[Dict[str, Any]]:
        # 单一收口：所有构造 OnlineBackend 的地方（15 处）都经过这里，禁用时
        # 立刻返回 None，不发请求、不等超时。
        if online_services_disabled():
            warn_online_disabled_once(service_name)
            return None
        if not self._base:
            raise RuntimeError("Online access requires AUDIT_ONLINE_SERVICE_BASE")
        try:
            url = f"{self._base}/{service_name}"
            body = {"params": params}
            resp = self._post(url, body)
            if not resp.get("success"):
                logger.warning(f"OnlineBackend: {service_name} returned success=false")
                return None
            datas = resp.get("data", {}).get("datas")
            if datas is None:
                return {}
            if not isinstance(datas, dict):
                return datas
            return {k: _maybe_json(v) for k, v in datas.items()}
        except Exception as e:
            logger.warning(f"OnlineBackend: {service_name} failed: {e}")
            return None


class OfflineBackend(DataBackend):
    """离线模式：从 calls_blob 字符串解析服务返回。

    calls_blob 格式: 多行，每行 \\x01 分隔：
        [0]时间戳 [1]ServiceName [2]snapKey [3]... [4]来源 [5]traceId [6]时间戳2 [7]response_json
    response_json 结构: {"response": {...}, "result": {success/code}}

    同一个服务可能被调用多次（如 queryNoteInfo 既查待审笔记又查相似笔记），
    通过 note_id 参数指定待审笔记 ID，优先返回匹配该 note_id 的调用结果。
    """

    def __init__(self, calls_blob: str, note_id: str = ""):
        self._note_id = note_id
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._parse(calls_blob)

    def _parse(self, calls_blob: str):
        for line in calls_blob.split("\n"):
            parts = line.split("\x01")
            if len(parts) < 8:
                continue
            svc = parts[1]
            try:
                raw = json.loads(parts[7])
                response = raw.get("response", raw)
                if isinstance(response, dict):
                    parsed = {k: _maybe_json(v) for k, v in response.items()}
                else:
                    parsed = response
                if svc not in self._cache:
                    self._cache[svc] = []
                self._cache[svc].append(parsed)
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

    def call(self, service_name: str, **params) -> Optional[Dict[str, Any]]:
        entries = self._cache.get(service_name)
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0]

        # 多条结果时，按 params 匹配
        # 优先用 uid/userId 匹配（GetUserInfoAll 等用户服务）
        target_uid = params.get("uid", params.get("userId", ""))
        if target_uid:
            for entry in entries:
                if self._entry_matches_user_id(entry, target_uid):
                    return entry

        # 其次用 noteId 匹配
        target_nid = params.get("noteId", params.get("noteIds", [None]))
        if isinstance(target_nid, list):
            target_nid = target_nid[0] if target_nid else None
        target_nid = target_nid or self._note_id

        if target_nid:
            for entry in entries:
                if self._entry_matches_note_id(entry, target_nid):
                    return entry

        # fallback: 返回第一条
        return entries[0]

    @staticmethod
    def _entry_matches_note_id(entry: Dict[str, Any], note_id: str) -> bool:
        """检查一条服务返回是否匹配指定 note_id。"""
        if not isinstance(entry, dict):
            return False
        # queryNoteInfo: noteInfo.noteDetailInfo.noteId
        note_info = entry.get("noteInfo", {})
        if isinstance(note_info, dict):
            detail = note_info.get("noteDetailInfo", {})
            if isinstance(detail, dict) and detail.get("noteId") == note_id:
                return True
        # 直接顶层 noteId
        if entry.get("noteId") == note_id:
            return True
        # noteAndType key 匹配
        nat = entry.get("noteAndType", {})
        if isinstance(nat, dict) and note_id in nat:
            return True
        return False

    @staticmethod
    def _entry_matches_user_id(entry: Dict[str, Any], user_id: str) -> bool:
        """检查一条服务返回是否匹配指定 user_id。"""
        if not isinstance(entry, dict):
            return False
        if entry.get("userId") == user_id:
            return True
        if entry.get("uid") == user_id:
            return True
        return False
