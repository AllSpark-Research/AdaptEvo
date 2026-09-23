# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from typing import Any, Dict, Optional

from loguru import logger

from relax.utils.metrics.client import get_metrics_client
from relax.utils.timer import Timer
from relax.utils.utils import get_serve_url


# Special key to mark timeline events in metrics
TIMELINE_EVENTS_KEY = "__timeline_events__"


class MetricsServiceAdapter:
    def __init__(self, args: Namespace):
        self.args = args

        # Get service URL from get_serve_url with /metrics route prefix
        actual_service_url = get_serve_url(route_prefix="/metrics")

        self.client = get_metrics_client(actual_service_url)

        health = self.client.health_check()
        if health.get("status") == "healthy":
            logger.info(f"MetricsServiceAdapter: Connected to metrics service at {actual_service_url}")
        else:
            logger.warning(
                f"MetricsServiceAdapter: Warning - Cannot connect to metrics service: {health.get('message')}"
            )

        # Check if timeline tracing is enabled
        self._timeline_enabled = getattr(args, "timeline_dump_dir", None) is not None
        if self._timeline_enabled:
            logger.info(f"MetricsServiceAdapter: Timeline tracing enabled, dump dir: {args.timeline_dump_dir}")

    def log(self, metrics: Dict[str, Any], step_key: str = "step") -> bool:
        if step_key not in metrics:
            logger.error(f"MetricsServiceAdapter: Error - step_key '{step_key}' not found in metrics")
            return False

        step = metrics[step_key]
        if not isinstance(step, int):
            logger.warning(f"MetricsServiceAdapter: Error - step value must be int, got {type(step)}")
            return False

        metrics_to_send = {k: v for k, v in metrics.items() if k != step_key}

        # Add timeline events to metrics if enabled
        if self._timeline_enabled:
            timer = Timer()
            records = timer.log_record_and_clear(step=step)
            if records:
                # Convert records to trace events and add to metrics
                events = [record.to_trace_event() for record in records]
                metrics_to_send[TIMELINE_EVENTS_KEY] = events

        if not metrics_to_send:
            logger.warning(f"MetricsServiceAdapter: Warning - No metrics to send for step {step}")
            return True

        result = self.client.log_metrics_batch(step, metrics_to_send, immediate=True)
        if not result:
            logger.error(f"MetricsServiceAdapter: Failed to log metrics for step {step}")
            return False

        return True

    def flush(self, step) -> bool:
        report_result = self.client.report_step(step)
        if report_result.get("status") != "success":
            logger.error(f"MetricsServiceAdapter: Failed to report step {step}: {report_result.get('message')}")
            return False

        return True

    def direct_log(self, step: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Add timeline events if enabled
        if self._timeline_enabled:
            timer = Timer()
            records = timer.log_record_and_clear(step=step)
            if records:
                events = [record.to_trace_event() for record in records]
                metrics = metrics.copy()
                metrics[TIMELINE_EVENTS_KEY] = events

        result = self.client.log_metrics_batch(step, metrics, immediate=False)
        if not result:
            logger.error(f"MetricsServiceAdapter: Failed to log metrics for step {step}")
            return {"status": "error", "message": f"Failed to log metrics batch for step {step}"}
        return self.client.report_step(step)


_global_adapter: Optional[MetricsServiceAdapter] = None


def init_metrics_service_adapter(args: Namespace) -> MetricsServiceAdapter:
    """Initialize the global metrics service adapter.

    The service URL is automatically determined using get_serve_url() with
    the /metrics route prefix.

    Args:
        args: Configuration arguments

    Returns:
        MetricsServiceAdapter: The global adapter instance
    """
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = MetricsServiceAdapter(args)
    return _global_adapter


def get_metrics_service_adapter() -> Optional[MetricsServiceAdapter]:
    return _global_adapter
