# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TimelineTrace Adapter for Chrome Trace Event format."""

import json
import os
from typing import Any, Dict, List

from relax.utils.timer import TimelineEvent


class TimelineTraceAdapter:
    """Adapter for generating Chrome Timeline Trace files.

    This adapter receives TimelineEvent records and generates Chrome Trace
    Event format JSON files that can be visualized in chrome://tracing or
    Perfetto.
    """

    def __init__(self, dump_dir: str, max_dump: int = 20):
        """Initialize the TimelineTraceAdapter.

        Args:
            dump_dir: Directory to dump timeline trace files. If None or empty,
                     the adapter is disabled.
        """
        self.dump_dir = dump_dir
        self.enabled = bool(dump_dir and dump_dir.strip())
        self._all_events: List[Dict[str, Any]] = []
        self._dump_cnt = 0
        self._max_dump = max_dump

        if self.enabled:
            # Ensure the directory exists
            os.makedirs(dump_dir, exist_ok=True)

    def is_enabled(self) -> bool:
        """Check if the adapter is enabled."""
        return self.enabled

    def add_events(self, events: List[TimelineEvent]):
        """Add TimelineEvent records.

        Args:
            events: List of TimelineEvent objects to add.
        """
        if not self.enabled:
            return

        for event in events:
            trace_event = event.to_trace_event()
            self._all_events.append(trace_event)

    def add_event_dicts(self, event_dicts: List[Dict[str, Any]]):
        """Add already-serialized event dictionaries.

        Args:
            event_dicts: List of event dictionaries in TimelineEvent format.
        """
        if not self.enabled:
            return

        self._all_events.extend(event_dicts)

    def dump(self, step: int):
        """Dump all collected events to a JSON file.

        The file is written to {dump_dir}/timeline_step_{step}.json

        Args:
            step: The current step number.
        """
        if not self.enabled:
            return

        if not self._all_events:
            return

        if self._dump_cnt >= self._max_dump:
            return

        # Sort events by timestamp
        sorted_events = sorted(self._all_events, key=lambda e: e.get("ts", 0))

        filename = f"timeline_step_{step}.json"
        filepath = os.path.join(self.dump_dir, filename)

        with open(filepath, "w") as f:
            json.dump(sorted_events, f)

        self._dump_cnt += 1

        # DO NOT CLEAR
        # self._all_events.clear()

    def dump_all(self, step: int):
        """Alias for dump() for backward compatibility."""
        self.dump(step)

    def clear(self):
        """Clear all stored events without dumping."""
        self._all_events.clear()

    def get_event_count(self) -> int:
        """Get the number of events currently stored."""
        return len(self._all_events)
