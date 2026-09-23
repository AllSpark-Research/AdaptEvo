# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metrics collection module for autoscaler.

This module provides functionality to collect, parse, and aggregate metrics
from SGLang inference engines. It supports Prometheus-format metrics and
computes aggregated statistics for scaling decisions.
"""

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.utils.autoscaler.config import AutoscalerConfig

logger = get_logger(__name__)


@dataclass
class EngineMetrics:
    """Single engine metrics snapshot.

    This dataclass holds a snapshot of metrics from a single SGLang engine
    at a point in time. It includes both primary scaling metrics and
    auxiliary metrics for observability.

    Attributes:
        engine_url: URL of the engine (e.g., "http://localhost:30000").
        engine_id: Unique identifier for the engine.
        timestamp: Unix timestamp when metrics were collected.
        token_usage: KV cache token usage ratio (0.0 - 1.0).
        num_queue_reqs: Number of requests waiting in queue.
        num_running_reqs: Number of requests currently being processed.
        gen_throughput: Generation throughput in tokens/second.
        max_total_num_tokens: Maximum total tokens in KV cache pool.
        num_used_tokens: Currently used tokens in KV cache.
        queue_time_p95: P95 queue waiting time in seconds.
        ttft_p95: P95 time-to-first-token in seconds.
        itl_p95: P95 inter-token latency in seconds.
        e2e_latency_p95: P95 end-to-end request latency in seconds.
        num_prefill_prealloc_queue_reqs: Requests in prefill prealloc queue.
        num_prefill_inflight_queue_reqs: Requests in prefill inflight queue.
        num_decode_prealloc_queue_reqs: Requests in decode prealloc queue.
        num_decode_transfer_queue_reqs: Requests in decode transfer queue.
    """

    engine_url: str
    engine_id: str
    timestamp: float

    # Primary scaling metrics
    token_usage: float = 0.0
    num_queue_reqs: int = 0
    num_running_reqs: int = 0
    gen_throughput: float = 0.0

    # Resource metrics
    max_total_num_tokens: int = 0
    num_used_tokens: int = 0

    # Latency metrics (from histogram quantiles)
    queue_time_p95: float = 0.0
    ttft_p95: float = 0.0
    itl_p95: float = 0.0
    e2e_latency_p95: float = 0.0

    # Detailed queue metrics
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_inflight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "engine_url": self.engine_url,
            "engine_id": self.engine_id,
            "timestamp": self.timestamp,
            "token_usage": self.token_usage,
            "num_queue_reqs": self.num_queue_reqs,
            "num_running_reqs": self.num_running_reqs,
            "gen_throughput": self.gen_throughput,
            "max_total_num_tokens": self.max_total_num_tokens,
            "num_used_tokens": self.num_used_tokens,
            "queue_time_p95": self.queue_time_p95,
            "ttft_p95": self.ttft_p95,
            "itl_p95": self.itl_p95,
            "e2e_latency_p95": self.e2e_latency_p95,
            "num_prefill_prealloc_queue_reqs": self.num_prefill_prealloc_queue_reqs,
            "num_prefill_inflight_queue_reqs": self.num_prefill_inflight_queue_reqs,
            "num_decode_prealloc_queue_reqs": self.num_decode_prealloc_queue_reqs,
            "num_decode_transfer_queue_reqs": self.num_decode_transfer_queue_reqs,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all engines.

    This dataclass holds aggregated statistics computed from individual
    engine metrics, suitable for scaling decisions.

    Attributes:
        num_engines: Total number of engines reporting metrics.
        total_queue_reqs: Sum of queue requests across all engines.
        total_running_reqs: Sum of running requests across all engines.
        avg_token_usage: Average token usage ratio across engines.
        total_throughput: Total generation throughput in tokens/second.
        max_queue_time_p95: Maximum P95 queue time across engines.
        max_ttft_p95: Maximum P95 TTFT across engines.
        max_itl_p95: Maximum P95 ITL across engines.
        throughput_variance: Relative variance in throughput over time window.
        timestamp: Unix timestamp of aggregation.
    """

    num_engines: int = 0
    total_queue_reqs: int = 0
    total_running_reqs: int = 0
    avg_token_usage: float = 0.0
    total_throughput: float = 0.0
    max_queue_time_p95: float = 0.0
    max_ttft_p95: float = 0.0
    max_itl_p95: float = 0.0
    throughput_variance: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_engines": self.num_engines,
            "total_queue_reqs": self.total_queue_reqs,
            "total_running_reqs": self.total_running_reqs,
            "avg_token_usage": self.avg_token_usage,
            "total_throughput": self.total_throughput,
            "max_queue_time_p95": self.max_queue_time_p95,
            "max_ttft_p95": self.max_ttft_p95,
            "max_itl_p95": self.max_itl_p95,
            "throughput_variance": self.throughput_variance,
            "timestamp": self.timestamp,
        }


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus text format into a metric dictionary.

    This function parses the Prometheus exposition format and extracts
    metric values into a flat dictionary. For metrics with the same name
    but different labels (e.g., from different tp_rank/pp_rank), values
    are aggregated (summed for gauges, last value for counters).

    Args:
        text: Raw Prometheus metrics text.

    Returns:
        Dictionary mapping metric names to aggregated values.
    """
    metrics: Dict[str, float] = {}
    # Track metrics that appear multiple times (different labels)
    metric_counts: Dict[str, int] = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse: metric_name{labels} value or metric_name value
        # Match the metric name and optional labels
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)((?:\{[^}]*\})?)?\s+(.+)$", line)
        if not match:
            continue

        name = match.group(1)
        value_str = match.group(3)

        try:
            value = float(value_str)

            # For metrics that appear multiple times (different labels),
            # aggregate by summing values (appropriate for gauges like num_running_reqs)
            if name in metrics:
                # If we've seen this metric before, sum the values
                metrics[name] += value
                metric_counts[name] += 1
            else:
                metrics[name] = value
                metric_counts[name] = 1
        except ValueError:
            continue

    # Log aggregation info for debugging
    aggregated_metrics = {k: v for k, v in metric_counts.items() if v > 1}
    if aggregated_metrics:
        logger.debug(f"Aggregated metrics with multiple label combinations: {aggregated_metrics}")

    return metrics


def extract_histogram_quantile(
    metrics: Dict[str, float],
    metric_name: str,
    quantile: float,
    count_key: str,
) -> float:
    """Extract approximate quantile from Prometheus histogram buckets.

    This function computes an approximate quantile value from histogram
    bucket cumulative counts. It uses linear interpolation within buckets.

    Args:
        metrics: Dictionary of metric name to value.
        metric_name: Base name of the histogram metric (e.g., "sglang:queue_time_seconds").
        quantile: Target quantile (e.g., 0.95 for P95).
        count_key: Key for the total count metric.

    Returns:
        Approximate quantile value, or 0.0 if not available.
    """
    total_count = metrics.get(count_key, 0)
    if total_count == 0:
        return 0.0

    # Collect bucket boundaries and cumulative counts
    buckets: List[tuple] = []
    bucket_pattern = re.compile(rf"^{re.escape(metric_name)}_bucket.*le=\"([0-9.+eE]+)\"")

    for key, value in metrics.items():
        if key.startswith(f"{metric_name}_bucket"):
            match = bucket_pattern.search(key)
            if match:
                le = float(match.group(1))
                buckets.append((le, float(value)))

    if not buckets:
        return 0.0

    # Sort by bucket boundary
    buckets.sort(key=lambda x: x[0])

    # Find the bucket containing the target quantile
    target_count = total_count * quantile

    prev_count = 0.0
    prev_le = 0.0

    for le, cumulative_count in buckets:
        if cumulative_count >= target_count:
            # Linear interpolation within bucket
            if cumulative_count == prev_count:
                return le  # Empty bucket, use boundary

            bucket_range = le - prev_le
            count_in_bucket = cumulative_count - prev_count
            count_needed = target_count - prev_count

            if count_in_bucket > 0:
                interpolation = count_needed / count_in_bucket
                return prev_le + bucket_range * interpolation
            return le

        prev_count = cumulative_count
        prev_le = le

    # Quantile exceeds all buckets
    return buckets[-1][0]


class MetricsCollector:
    """Collects metrics from SGLang engines periodically.

    This class manages the collection of metrics from multiple engines,
    maintains a history of metrics snapshots, and provides aggregated
    statistics for scaling decisions.

    Attributes:
        config: AutoscalerConfig instance.
        _history: Deque of historical metrics snapshots.
        _session: aiohttp ClientSession for HTTP requests.
    """

    def __init__(self, config: "AutoscalerConfig"):
        """Initialize the metrics collector.

        Args:
            config: Autoscaler configuration.
        """
        self.config = config
        history_size = max(
            1,
            int(config.condition_window_secs / config.metrics_interval_secs),
        )
        self._history: deque = deque(maxlen=history_size)
        self._session: Optional[Any] = None  # aiohttp.ClientSession

    async def start(self) -> None:
        """Initialize async resources (HTTP session)."""
        try:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0),
            )
            logger.info("MetricsCollector started")
        except ImportError:
            logger.error("aiohttp not installed, metrics collection will fail")
            raise

    async def stop(self) -> None:
        """Clean up async resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        logger.info("MetricsCollector stopped")

    async def collect_from_engine(
        self,
        engine_url: str,
        engine_id: str,
    ) -> Optional[EngineMetrics]:
        """Fetch and parse metrics from a single engine.

        Args:
            engine_url: Base URL of the engine.
            engine_id: Unique identifier for the engine.

        Returns:
            EngineMetrics snapshot, or None if collection failed.
        """
        if self._session is None:
            logger.warning("HTTP session not initialized")
            return None

        metrics_url = f"{engine_url.rstrip('/')}/metrics"

        try:
            async with self._session.get(metrics_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch metrics from {engine_url}: HTTP {response.status}")
                    return None

                text = await response.text()
                raw_metrics = parse_prometheus_metrics(text)

                # Debug log key metrics for troubleshooting
                logger.debug(
                    f"Collected metrics from {engine_id}: "
                    f"num_running_reqs={raw_metrics.get('sglang:num_running_reqs', 'N/A')}, "
                    f"num_queue_reqs={raw_metrics.get('sglang:num_queue_reqs', 'N/A')}, "
                    f"token_usage={raw_metrics.get('sglang:token_usage', 'N/A'):.3f}"
                    if "sglang:token_usage" in raw_metrics
                    else "token_usage=N/A"
                )

                # Extract histogram quantiles
                queue_time_p95 = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:queue_time_seconds",
                    0.95,
                    "sglang:queue_time_seconds_count",
                )
                ttft_p95 = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:time_to_first_token_seconds",
                    0.95,
                    "sglang:time_to_first_token_seconds_count",
                )
                itl_p95 = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:inter_token_latency_seconds",
                    0.95,
                    "sglang:inter_token_latency_seconds_count",
                )
                e2e_p95 = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:e2e_request_latency_seconds",
                    0.95,
                    "sglang:e2e_request_latency_seconds_count",
                )

                return EngineMetrics(
                    engine_url=engine_url,
                    engine_id=engine_id,
                    timestamp=time.time(),
                    token_usage=raw_metrics.get("sglang:token_usage", 0.0),
                    num_queue_reqs=int(raw_metrics.get("sglang:num_queue_reqs", 0)),
                    num_running_reqs=int(raw_metrics.get("sglang:num_running_reqs", 0)),
                    gen_throughput=raw_metrics.get("sglang:gen_throughput", 0.0),
                    max_total_num_tokens=int(raw_metrics.get("sglang:max_total_num_tokens", 0)),
                    num_used_tokens=int(raw_metrics.get("sglang:num_used_tokens", 0)),
                    queue_time_p95=queue_time_p95,
                    ttft_p95=ttft_p95,
                    itl_p95=itl_p95,
                    e2e_latency_p95=e2e_p95,
                    num_prefill_prealloc_queue_reqs=int(raw_metrics.get("sglang:num_prefill_prealloc_queue_reqs", 0)),
                    num_prefill_inflight_queue_reqs=int(raw_metrics.get("sglang:num_prefill_inflight_queue_reqs", 0)),
                    num_decode_prealloc_queue_reqs=int(raw_metrics.get("sglang:num_decode_prealloc_queue_reqs", 0)),
                    num_decode_transfer_queue_reqs=int(raw_metrics.get("sglang:num_decode_transfer_queue_reqs", 0)),
                )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching metrics from {engine_url}")
            return None
        except Exception as e:
            logger.warning(f"Error collecting metrics from {engine_url}: {e}")
            return None

    async def collect_all(
        self,
        engines: List[Dict[str, str]],
    ) -> Dict[str, EngineMetrics]:
        """Collect metrics from all engines concurrently.

        Args:
            engines: List of engine info dicts with 'id' and 'url' keys.

        Returns:
            Dictionary mapping engine_id to EngineMetrics.
        """
        if not engines:
            return {}

        tasks = [self.collect_from_engine(engine["url"], engine["id"]) for engine in engines]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics_dict: Dict[str, EngineMetrics] = {}
        for engine, result in zip(engines, results):
            if isinstance(result, BaseException):
                logger.warning(f"Exception collecting from {engine['id']}: {result}")
            elif result is not None:
                metrics_dict[result.engine_id] = result

        return metrics_dict

    def add_snapshot(self, snapshot: Dict[str, EngineMetrics]) -> None:
        """Add a metrics snapshot to history.

        Args:
            snapshot: Dictionary mapping engine_id to EngineMetrics.
        """
        self._history.append(
            {
                "timestamp": time.time(),
                "metrics": snapshot,
            }
        )

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Compute aggregated metrics from recent history.

        Returns:
            AggregatedMetrics instance with computed statistics.
        """
        if not self._history:
            return AggregatedMetrics()

        # Get latest snapshot
        latest = self._history[-1]["metrics"]

        if not latest:
            return AggregatedMetrics()

        # Aggregate across engines
        num_engines = len(latest)
        total_queue = sum(m.num_queue_reqs for m in latest.values())
        total_running = sum(m.num_running_reqs for m in latest.values())
        total_throughput = sum(m.gen_throughput for m in latest.values())
        avg_token_usage = sum(m.token_usage for m in latest.values()) / num_engines if num_engines > 0 else 0.0

        # Compute max P95 latencies
        all_queue_times = [m.queue_time_p95 for m in latest.values() if m.queue_time_p95 > 0]
        all_ttft = [m.ttft_p95 for m in latest.values() if m.ttft_p95 > 0]
        all_itl = [m.itl_p95 for m in latest.values() if m.itl_p95 > 0]

        max_queue_time_p95 = max(all_queue_times) if all_queue_times else 0.0
        max_ttft_p95 = max(all_ttft) if all_ttft else 0.0
        max_itl_p95 = max(all_itl) if all_itl else 0.0

        # Compute throughput variance over time window
        throughput_variance = 0.0
        if len(self._history) >= 3:
            throughputs = [sum(m.gen_throughput for m in h["metrics"].values()) for h in list(self._history)[-10:]]
            if throughputs:
                mean_t = sum(throughputs) / len(throughputs)
                if mean_t > 0:
                    max_deviation = max(abs(t - mean_t) for t in throughputs)
                    throughput_variance = max_deviation / mean_t

        return AggregatedMetrics(
            num_engines=num_engines,
            total_queue_reqs=total_queue,
            total_running_reqs=total_running,
            avg_token_usage=avg_token_usage,
            total_throughput=total_throughput,
            max_queue_time_p95=max_queue_time_p95,
            max_ttft_p95=max_ttft_p95,
            max_itl_p95=max_itl_p95,
            throughput_variance=throughput_variance,
            timestamp=time.time(),
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the metrics history for debugging/observability.

        Returns:
            List of historical snapshots.
        """
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the metrics history."""
        self._history.clear()

    def get_condition_duration(
        self,
        condition_name: str,
        check_fn: Callable[[Dict[str, EngineMetrics]], bool],
    ) -> float:
        """Check how long a condition has been continuously true.

        Args:
            condition_name: Name of the condition (for logging).
            check_fn: Function that takes a metrics snapshot and returns True
                if the condition is met.

        Returns:
            Duration in seconds that the condition has been continuously true.
        """
        duration = 0.0
        interval = self.config.metrics_interval_secs

        for snapshot in reversed(list(self._history)):
            if check_fn(snapshot["metrics"]):
                duration += interval
            else:
                break

        return duration
