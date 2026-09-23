"""Shared switch for cache-backed versus live audit data access.

The default deliberately preserves the existing training/eval behavior.  The
online mode is opt-in because it can call production rule and tool services.
"""

from __future__ import annotations

import os


CACHE_DATA_ACCESS_MODE = "cache"
ONLINE_DATA_ACCESS_MODE = "online"
CACHE_RULE_ACCESS_MODE = "cache"
ONLINE_RULE_ACCESS_MODE = "online"


def data_access_mode() -> str:
    """Return the normalized audit data-access mode.

    Unknown values fall back to ``cache`` so a typo cannot unexpectedly turn
    on production service traffic during rollout.
    """
    value = os.environ.get("AUDIT_DATA_ACCESS_MODE", CACHE_DATA_ACCESS_MODE)
    value = str(value or "").strip().lower()
    return value if value in {CACHE_DATA_ACCESS_MODE, ONLINE_DATA_ACCESS_MODE} else CACHE_DATA_ACCESS_MODE


def online_data_access_enabled() -> bool:
    return data_access_mode() == ONLINE_DATA_ACCESS_MODE


def rule_access_mode() -> str:
    """Return whether rules are read from local artifacts or Jupiter.

    Rule access is intentionally independent from tool/data access. This lets
    production evidence tools stay live while stable rule text is loaded from
    the generated local cache.
    """
    value = os.environ.get("AUDIT_RULE_ACCESS_MODE", "")
    value = str(value or "").strip().lower()
    if value in {CACHE_RULE_ACCESS_MODE, ONLINE_RULE_ACCESS_MODE}:
        return value
    return data_access_mode()


def online_rule_access_enabled() -> bool:
    return rule_access_mode() == ONLINE_RULE_ACCESS_MODE
