from __future__ import annotations

from audit_agentic.environment import tool_result_cache


def test_note_history_identity_ignores_runtime_args_and_env(monkeypatch):
    monkeypatch.setenv("AUDIT_TOOL_CACHE_VERSION", "human-cot-0818-v1")
    monkeypatch.setenv("AUDIT_TOOL_CACHE_IDENTITY_MODE", "note_history")
    monkeypatch.setenv("BLOB_INDEX_PATH", "/runtime/blob.pkl")
    monkeypatch.setenv("IMAGE_CACHE_DIR", "/runtime/images")
    monkeypatch.setenv("RECENT_NOTES_MAX_NOTES", "5")

    payload = tool_result_cache.cache_key_payload(
        "get_recent_notes",
        {
            "note_id": "note-1",
            "history_id": "history-1",
            "source": "source-a",
            "limit": 10,
            "_audit_context": {"source": "source-a"},
        },
    )

    assert payload["args"] == {
        "note_id": "note-1",
        "history_id": "history-1",
    }
    assert payload["env"] == {
        key: "" for key in tool_result_cache._CACHE_ENV_KEYS
    }


def test_default_identity_keeps_semantic_args_and_env(monkeypatch):
    monkeypatch.setenv("AUDIT_TOOL_CACHE_IDENTITY_MODE", "default")
    monkeypatch.setenv("IMAGE_CACHE_DIR", "/runtime/images")

    payload = tool_result_cache.cache_key_payload(
        "get_recent_notes",
        {
            "note_id": "note-1",
            "history_id": "history-1",
            "limit": 10,
            "_audit_context": {"ignored": True},
        },
    )

    assert payload["args"] == {
        "note_id": "note-1",
        "history_id": "history-1",
        "limit": 10,
    }
    assert payload["env"]["IMAGE_CACHE_DIR"] == "/runtime/images"
