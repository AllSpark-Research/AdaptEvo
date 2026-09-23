"""Preserve exact Relax wire history across AgentScope ReAct turns."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


_IGNORED_MESSAGE_KEYS = {
    "audio",
    "function_call",
    "name",
    "reasoning",
    "reasoning_content",
    "refusal",
}
_IGNORED_TOOL_CALL_KEYS = {"index"}


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def _content_parts(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        return [{"type": "opaque", "value": value}]

    parts: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            parts.append({"type": "opaque", "value": item})
            continue

        part_type = str(item.get("type") or "")
        if part_type in {"text", "input_text", "output_text"}:
            parts.append({"type": "text", "text": str(item.get("text") or "")})
            continue
        if part_type in {"image", "image_url", "input_image"}:
            image_value = item.get("image_url", item.get("url", item.get("image")))
            if isinstance(image_value, dict):
                image_value = image_value.get("url")
            parts.append({"type": "image", "url": image_value})
            continue
        parts.append({"type": part_type or "opaque", "value": item})
    return parts


def _tool_calls_equivalent(
    current: Any,
    stored: Any,
    id_map: dict[str, str],
    tool_name_aliases: set[tuple[str, str]],
) -> bool:
    current_calls = list(current or [])
    stored_calls = list(stored or [])
    if len(current_calls) != len(stored_calls):
        return False

    for current_call, stored_call in zip(current_calls, stored_calls):
        if not isinstance(current_call, dict) or not isinstance(stored_call, dict):
            return False
        if current_call.get("type", "function") != stored_call.get(
            "type",
            "function",
        ):
            return False

        current_function = current_call.get("function") or {}
        stored_function = stored_call.get("function") or {}
        if not isinstance(current_function, dict) or not isinstance(
            stored_function,
            dict,
        ):
            return False
        current_name = str(current_function.get("name") or "")
        stored_name = str(stored_function.get("name") or "")
        if current_name != stored_name and (
            current_name,
            stored_name,
        ) not in tool_name_aliases:
            return False
        if _json_value(current_function.get("arguments", {})) != _json_value(
            stored_function.get("arguments", {}),
        ):
            return False

        current_id = current_call.get("id")
        stored_id = stored_call.get("id")
        if isinstance(current_id, str) and isinstance(stored_id, str):
            previous = id_map.get(current_id)
            if previous is not None and previous != stored_id:
                return False
            id_map[current_id] = stored_id
        elif current_id != stored_id:
            return False

        current_extra = {
            key: value
            for key, value in current_call.items()
            if key not in {"id", "type", "function"} | _IGNORED_TOOL_CALL_KEYS
        }
        stored_extra = {
            key: value
            for key, value in stored_call.items()
            if key not in {"id", "type", "function"} | _IGNORED_TOOL_CALL_KEYS
        }
        if current_extra != stored_extra:
            return False
    return True


def _messages_equivalent(
    current: dict[str, Any],
    stored: dict[str, Any],
    id_map: dict[str, str],
    tool_name_aliases: set[tuple[str, str]],
) -> bool:
    if current.get("role") != stored.get("role"):
        return False
    if _content_parts(current.get("content")) != _content_parts(stored.get("content")):
        return False
    if not _tool_calls_equivalent(
        current.get("tool_calls"),
        stored.get("tool_calls"),
        id_map,
        tool_name_aliases,
    ):
        return False

    current_tool_id = current.get("tool_call_id")
    stored_tool_id = stored.get("tool_call_id")
    if current_tool_id is not None or stored_tool_id is not None:
        if id_map.get(str(current_tool_id), current_tool_id) != stored_tool_id:
            return False

    semantic_keys = (set(current) | set(stored)) - {
        "content",
        "role",
        "tool_call_id",
        "tool_calls",
    } - _IGNORED_MESSAGE_KEYS
    return all(current.get(key) == stored.get(key) for key in semantic_keys)


def _rewrite_tool_call_ids(value: Any, id_map: dict[str, str]) -> Any:
    if not id_map:
        return value
    if isinstance(value, list):
        return [_rewrite_tool_call_ids(item, id_map) for item in value]
    if not isinstance(value, dict):
        return value

    rewritten = {
        key: _rewrite_tool_call_ids(item, id_map)
        for key, item in value.items()
    }
    tool_call_id = rewritten.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id in id_map:
        rewritten["tool_call_id"] = id_map[tool_call_id]
    for tool_call in rewritten.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if isinstance(call_id, str) and call_id in id_map:
            tool_call["id"] = id_map[call_id]
    return rewritten


def _assistant_wire_message(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return None
    message = copy.deepcopy(choice["message"])
    message.setdefault("role", "assistant")
    return message


@dataclass
class RelaxWireHistory:
    """Trajectory-local exact messages observed by the Relax proxy."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_name_aliases: set[tuple[str, str]] = field(default_factory=set)
    canonicalized_turns: int = 0
    rejected_turns: int = 0

    def add_tool_name_alias(self, resolved: str, raw: str) -> None:
        if resolved and raw and resolved != raw:
            self.tool_name_aliases.add((resolved, raw))

    def reconcile(self, rebuilt_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rebuilt = copy.deepcopy(list(rebuilt_messages or []))
        if not self.messages:
            return rebuilt
        if len(rebuilt) < len(self.messages):
            self.rejected_turns += 1
            return rebuilt

        id_map: dict[str, str] = {}
        for current, stored in zip(rebuilt[: len(self.messages)], self.messages):
            if not isinstance(current, dict) or not _messages_equivalent(
                current,
                stored,
                id_map,
                self.tool_name_aliases,
            ):
                self.rejected_turns += 1
                return rebuilt

        tail = _rewrite_tool_call_ids(rebuilt[len(self.messages) :], id_map)
        self.canonicalized_turns += 1
        return copy.deepcopy(self.messages) + tail

    def record(
        self,
        sent_messages: list[dict[str, Any]],
        response: dict[str, Any],
    ) -> None:
        history = copy.deepcopy(list(sent_messages or []))
        assistant = _assistant_wire_message(response)
        if assistant is not None:
            history.append(assistant)
        self.messages = history
