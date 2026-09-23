# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metrics package for Relax.

This package provides:
    - MetricsService: Ray Serve deployment for centralized metrics collection and reporting
    - MetricsClient: HTTP client for interacting with the Metrics Service
    - MetricsBuffer: Thread-safe buffer for step-based metric aggregation
    - TimelineTraceAdapter: Adapter for Chrome Timeline Trace format

Service side:
    - relax.metrics.service: MetricsService (Ray Serve deployment)

Client side:
    - relax.metrics.client: MetricsClient, get_metrics_client

Adapters:
    - relax.metrics.timeline_trace: TimelineTraceAdapter
"""

from relax.utils.metrics.client import MetricsClient, get_metrics_client
from relax.utils.metrics.timeline_trace import TimelineTraceAdapter


__all__ = [
    "MetricsClient",
    "get_metrics_client",
    "TimelineTraceAdapter",
]
