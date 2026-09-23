"""Protocol-aware AgentScope OpenAI model for Relax-managed rollouts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolChoice

from .relax_wire_history import RelaxWireHistory


class ProtocolAwareOpenAIChatModel(OpenAIChatModel):
    """Repair conservative tool-name typos while preserving trainable history."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.stream:
            raise ValueError("ProtocolAwareOpenAIChatModel requires stream=False")
        self._turn_index = 0
        self._wire_history = RelaxWireHistory()
        self._protocol_events: list[dict[str, Any]] = []

    @property
    def model_call_count(self) -> int:
        return int(self._turn_index)

    @property
    def wire_history_stats(self) -> dict[str, int]:
        return {
            "message_count": len(self._wire_history.messages),
            "canonicalized_turns": self._wire_history.canonicalized_turns,
            "rejected_turns": self._wire_history.rejected_turns,
        }

    @property
    def protocol_diagnostics(self) -> dict[str, Any]:
        events = [dict(event) for event in self._protocol_events]
        return {
            "events": events,
            "protocol_error_count": len(events),
            "unknown_tool_call_count": sum(
                event["type"] == "unknown_tool_call" for event in events
            ),
            "tool_name_format_error_count": sum(
                event["type"] == "tool_name_format_error" for event in events
            ),
            "invalid_tool_arguments_count": sum(
                event["type"] == "invalid_tool_arguments" for event in events
            ),
            "repaired_tool_call_count": sum(
                bool(event.get("resolved_tool_name")) for event in events
            ),
        }

    async def _call_api(
        self,
        model_name: str,
        messages: list[Any],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        formatted_messages = await self.formatter.format(messages)
        formatted_messages = self._wire_history.reconcile(formatted_messages)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": False,
        }
        if self.parameters.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.parameters.max_tokens
        if self.parameters.temperature is not None:
            kwargs["temperature"] = self.parameters.temperature
        if self.parameters.top_p is not None:
            kwargs["top_p"] = self.parameters.top_p
        if self.parameters.thinking_enable and self.parameters.reasoning_effort:
            kwargs["reasoning_effort"] = self.parameters.reasoning_effort
        if self.extra_body is not None:
            kwargs["extra_body"] = dict(self.extra_body)
        kwargs.update(generate_kwargs)

        formatted_tools, formatted_choice = self._format_tools(tools, tool_choice)
        if formatted_tools:
            kwargs["tools"] = formatted_tools
            if not self.parameters.parallel_tool_calls:
                kwargs["parallel_tool_calls"] = False
        if formatted_choice is not None:
            kwargs["tool_choice"] = formatted_choice

        started_at = datetime.now()
        response = await self.client.chat.completions.create(**kwargs)
        raw_response = response.model_dump(mode="json")
        self._turn_index += 1
        repairs = self._record_protocol_events(
            raw_response,
            formatted_tools or [],
        )
        self._wire_history.record(formatted_messages, raw_response)
        self._apply_tool_name_repairs(raw_response, repairs)

        from openai.types.chat import ChatCompletion

        completion = ChatCompletion.model_validate(raw_response)
        return self._parse_completion_response(started_at, completion, "wav")

    @staticmethod
    def _canonical_tool_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(name or "").lower())

    @staticmethod
    def _formatted_tool_names(tools: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name:
                names.append(name)
        return names

    def _resolve_tool_name(
        self,
        name: str,
        available_names: list[str],
    ) -> tuple[str | None, str | None]:
        canonical = self._canonical_tool_name(name)
        if not canonical:
            return None, None

        exact_style_matches = [
            candidate
            for candidate in available_names
            if self._canonical_tool_name(candidate) == canonical
        ]
        if len(exact_style_matches) == 1:
            return exact_style_matches[0], "canonical"

        scored = sorted(
            (
                SequenceMatcher(
                    None,
                    canonical,
                    self._canonical_tool_name(candidate),
                ).ratio(),
                candidate,
            )
            for candidate in available_names
            if self._canonical_tool_name(candidate)
        )
        if not scored:
            return None, None
        best_score, best_name = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else 0.0
        threshold = float(
            os.environ.get("AUDIT_AGENTSCOPE_TOOL_NAME_MATCH_THRESHOLD", "0.84"),
        )
        margin = float(
            os.environ.get("AUDIT_AGENTSCOPE_TOOL_NAME_MATCH_MARGIN", "0.08"),
        )
        if best_score >= threshold and best_score - second_score >= margin:
            return best_name, "fuzzy"
        return None, None

    def _record_protocol_events(
        self,
        response: dict[str, Any],
        formatted_tools: list[dict[str, Any]],
    ) -> dict[tuple[int, int], str]:
        available_names = self._formatted_tool_names(formatted_tools)
        available_set = set(available_names)
        repairs: dict[tuple[int, int], str] = {}

        for choice_index, raw_choice in enumerate(response.get("choices") or []):
            choice = raw_choice if isinstance(raw_choice, dict) else {}
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for call_index, raw_call in enumerate(message.get("tool_calls") or []):
                call = raw_call if isinstance(raw_call, dict) else {}
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                name = str(function.get("name") or "").strip()
                base_event = {
                    "turn_index": self._turn_index,
                    "choice_index": choice_index,
                    "call_index": call_index,
                    "tool_call_id": str(call.get("id") or ""),
                    "tool_name": name,
                }

                if name not in available_set:
                    resolved_name, strategy = self._resolve_tool_name(
                        name,
                        available_names,
                    )
                    if resolved_name:
                        repairs[(choice_index, call_index)] = resolved_name
                        self._wire_history.add_tool_name_alias(resolved_name, name)
                        self._protocol_events.append(
                            {
                                **base_event,
                                "type": "tool_name_format_error",
                                "resolved_tool_name": resolved_name,
                                "repair_strategy": strategy,
                            },
                        )
                    else:
                        self._protocol_events.append(
                            {
                                **base_event,
                                "type": "unknown_tool_call",
                            },
                        )

                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        self._protocol_events.append(
                            {
                                **base_event,
                                "type": "invalid_tool_arguments",
                            },
                        )
        return repairs

    @staticmethod
    def _apply_tool_name_repairs(
        response: dict[str, Any],
        repairs: dict[tuple[int, int], str],
    ) -> None:
        for choice_index, choice in enumerate(response.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for call_index, call in enumerate(message.get("tool_calls") or []):
                repaired_name = repairs.get((choice_index, call_index))
                if not repaired_name or not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    function["name"] = repaired_name
