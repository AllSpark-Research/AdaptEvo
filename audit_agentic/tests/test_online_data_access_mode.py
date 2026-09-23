from __future__ import annotations

from audit_agentic.environment.data_access_mode import (
    CACHE_DATA_ACCESS_MODE,
    ONLINE_DATA_ACCESS_MODE,
    data_access_mode,
    online_data_access_enabled,
)
from audit_agentic.environment.services._base import online_services_disabled
from audit_agentic.environment.tool_result_cache import cache_disabled
from audit_agentic.environment.tools.image_cache import get_image_cache_dir


def test_data_access_mode_is_cache_by_default(monkeypatch):
    monkeypatch.delenv("AUDIT_DATA_ACCESS_MODE", raising=False)
    assert data_access_mode() == CACHE_DATA_ACCESS_MODE
    assert not online_data_access_enabled()


def test_unknown_data_access_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("AUDIT_DATA_ACCESS_MODE", "typo")
    assert data_access_mode() == CACHE_DATA_ACCESS_MODE


def test_online_mode_bypasses_local_caches_and_enables_services(monkeypatch):
    monkeypatch.setenv("AUDIT_DATA_ACCESS_MODE", ONLINE_DATA_ACCESS_MODE)
    monkeypatch.setenv("AUDIT_DISABLE_ONLINE_SERVICES", "1")
    monkeypatch.setenv("IMAGE_CACHE_DIR", "/stale/image/cache")

    assert cache_disabled()
    assert not online_services_disabled()
    assert get_image_cache_dir({"_image_cache_dir": "/another/stale/cache"}) == ""
