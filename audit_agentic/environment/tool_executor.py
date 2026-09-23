"""ToolExecutor: dispatch ToolCalls to registered runners.

Supports a *default_args* dict that's merged into every call's args (the
caller's args take precedence). This is how the workflow injects the
current note_id without forcing the LLM to fill it in.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..schemas import ToolCall, ToolObservation
from .tool_registry import DEFAULT_TOOL_REGISTRY
from .tool_result_cache import get_cached_result, set_cached_result
from .tools.image_cache import apply_image_cache


class ToolExecutor:
    def __init__(
        self,
        registry: Dict[str, Dict[str, Any]] | None = None,
        default_args: Optional[Dict[str, Any]] = None,
    ):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        self.default_args: Dict[str, Any] = default_args or {}

    def with_defaults(self, **defaults: Any) -> "ToolExecutor":
        """Return a child executor with merged default_args (for per-run binding)."""
        merged = {**self.default_args, **defaults}
        return ToolExecutor(registry=self.registry, default_args=merged)

    def _execute_one(self, call: ToolCall) -> ToolObservation:
        # Runtime-bound values are authoritative. Native tool calls must never
        # be able to switch the audited note/source by supplying reserved args.
        merged_args: Dict[str, Any] = {**(call.args or {}), **self.default_args}
        merged_call = call.model_copy(update={"args": merged_args})

        spec = self.registry.get(call.tool_name)
        if spec is None:
            return ToolObservation(
                tool_name=call.tool_name,
                status="failed",
                error="unknown_tool",
                call=merged_call,
            )
        runner = getattr(spec, "run", None)
        if runner is None:
            return ToolObservation(
                tool_name=call.tool_name,
                status="failed",
                error="no_runner_bound",
                call=merged_call,
            )
        cacheable = bool(getattr(spec, "cacheable", True))
        cached_result = get_cached_result(call.tool_name, merged_args) if cacheable else None
        if cached_result is not None:
            if isinstance(cached_result, dict):
                cached_result = apply_image_cache(cached_result, merged_args)
            return ToolObservation(
                tool_name=call.tool_name,
                status="ok",
                result=cached_result,
                call=merged_call,
            )
        try:
            result = runner(merged_args)
            if isinstance(result, dict):
                result = apply_image_cache(result, merged_args)
            if cacheable:
                set_cached_result(call.tool_name, merged_args, result)
            return ToolObservation(
                tool_name=call.tool_name,
                status="ok",
                result=result,
                call=merged_call,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolObservation(
                tool_name=call.tool_name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                call=merged_call,
            )

    def execute(self, tool_calls: List[ToolCall], concurrency: int = 1) -> List[ToolObservation]:
        workers = max(1, int(concurrency or 1))
        if workers == 1 or len(tool_calls) <= 1:
            return [self._execute_one(call) for call in tool_calls]

        indexed_results: List[tuple[int, ToolObservation]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(tool_calls))) as pool:
            futures = {pool.submit(self._execute_one, call): i for i, call in enumerate(tool_calls)}
            for fut in as_completed(futures):
                indexed_results.append((futures[fut], fut.result()))
        indexed_results.sort(key=lambda x: x[0])
        return [obs for _, obs in indexed_results]
