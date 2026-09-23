# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
Metrics Collector - Observability for DCS.

Provides:
- Performance metrics (latency, throughput)
- Memory usage tracking
- Prometheus-compatible export
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class MetricsSample:
    """A single metrics sample with value and timestamp.

    Used for time-series metrics collection and analysis.

    Attributes:
        value: Metric value (float)
        timestamp: Unix timestamp when sample was recorded
        labels: Optional tags/labels for metric categorization (e.g., {"role": "actor", "rank": "0"})
    """

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Histogram:
    """Simple histogram for latency tracking.

    Tracks observations in configurable buckets and computes statistics.
    Thread-safe for concurrent access from multiple workers.

    Default buckets (in seconds): [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    Example:
        hist = Histogram("request_latency_seconds")
        hist.observe(0.042)  # Record 42ms
        stats = hist.get_stats()
        print(f"Average latency: {stats['avg']:.3f}s")
    """

    DEFAULT_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, buckets: Optional[List[float]] = None):
        """Initialize histogram.

        Args:
            name: Histogram name (used for metrics export)
            buckets: List of bucket thresholds. If None, uses DEFAULT_BUCKETS
        """
        self.name = name
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts = defaultdict(int)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: The value to record (typically latency in seconds)
        """
        with self._lock:
            self._sum += value
            self._count += 1

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[bucket] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get histogram statistics.

        Returns:
            Dict with keys:
                - count: Number of observations
                - sum: Total sum of all observations
                - avg: Average (mean) value
                - buckets: Dict of {bucket_threshold: count_in_bucket}
        """
        with self._lock:
            if self._count == 0:
                return {"count": 0, "sum": 0, "avg": 0}

            return {
                "count": self._count,
                "sum": self._sum,
                "avg": self._sum / self._count,
                "buckets": dict(self._counts),
            }

    def reset(self) -> None:
        """Reset the histogram to zero state."""
        with self._lock:
            self._counts.clear()
            self._sum = 0.0
            self._count = 0


class Counter:
    """Simple thread-safe counter metric.

    Monotonically increasing counter for tracking total counts.
    Useful for tracking total operations, bytes transferred, errors, etc.

    Example:
        counter = Counter("requests_total")
        counter.inc()  # Increment by 1
        counter.inc(5)  # Increment by 5
        print(f"Total requests: {counter.get()}")
    """

    def __init__(self, name: str):
        """Initialize counter.

        Args:
            name: Counter name (used for metrics export)
        """
        self.name = name
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, value: int = 1) -> None:
        """Increment the counter.

        Args:
            value: Amount to increment by (default: 1)
        """
        with self._lock:
            self._value += value

    def get(self) -> int:
        """Get the current counter value.

        Returns:
            int: Current counter value
        """
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the counter to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """Simple thread-safe gauge metric.

    A gauge is a metric that can go up or down (unlike a counter which only increases).
    Useful for tracking current state like memory usage, active connections, etc.

    Example:
        gauge = Gauge("active_connections")
        gauge.set(42)  # Set to 42
        gauge.inc()    # Increment to 43
        gauge.dec(3)   # Decrement to 40
    """

    def __init__(self, name: str):
        """Initialize gauge.

        Args:
            name: Gauge name (used for metrics export)
        """
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge to a specific value.

        Args:
            value: Value to set
        """
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment the gauge.

        Args:
            value: Amount to increment by (default: 1.0)
        """
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement the gauge.

        Args:
            value: Amount to decrement by (default: 1.0)
        """
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get the current gauge value.

        Returns:
            float: Current gauge value
        """
        with self._lock:
            return self._value


class MetricsCollector:
    """Metrics collector for the Distributed Checkpoint Service.

    Tracks performance metrics including:
    - Operation latencies (save, load, send, recv) as histograms
    - Data volume transferred (bytes)
    - Operation counts
    - Memory and connection state

    Supports Prometheus-compatible export for integration with monitoring stacks.

    Tracked Metrics:
        Latencies (histogram):
        - dcs_save_latency_seconds: Time to save checkpoint
        - dcs_load_latency_seconds: Time to load checkpoint
        - dcs_send_latency_seconds: Time to send tensors over network
        - dcs_recv_latency_seconds: Time to receive tensors

        Counters (monotonic):
        - dcs_bytes_sent_total: Total bytes transmitted
        - dcs_bytes_received_total: Total bytes received
        - dcs_bytes_saved_total: Total bytes saved to storage
        - dcs_bytes_loaded_total: Total bytes loaded from storage
        - dcs_send_operations_total: Total send operations
        - dcs_recv_operations_total: Total receive operations
        - dcs_save_operations_total: Total save operations
        - dcs_load_operations_total: Total load operations
        - dcs_errors_total: Total errors encountered

        Gauges (can vary):
        - dcs_memory_buffer_usage_bytes: Current buffer memory in use
        - dcs_active_connections: Number of active P2P connections
        - dcs_pending_operations: Number of operations in flight
        - dcs_uptime_seconds: Time since collector started

    Example:
        metrics = MetricsCollector()

        # Record operations
        metrics.record_save(bytes_saved=1024*1024, duration=0.5)
        metrics.record_send(bytes_sent=512*1024, duration=0.1)

        # Query metrics
        all_metrics = metrics.get_all()
        print(f"Average send latency: {all_metrics['latency']['send']['avg']:.3f}s")

        # Export for Prometheus
        prom_text = metrics.export_prometheus()
    """

    def __init__(self):
        """Initialize the metrics collector."""
        # Latency histograms
        self.save_latency = Histogram("dcs_save_latency_seconds")
        self.load_latency = Histogram("dcs_load_latency_seconds")
        self.send_latency = Histogram("dcs_send_latency_seconds")
        self.recv_latency = Histogram("dcs_recv_latency_seconds")

        # Counters
        self.bytes_sent = Counter("dcs_bytes_sent_total")
        self.bytes_received = Counter("dcs_bytes_received_total")
        self.bytes_saved = Counter("dcs_bytes_saved_total")
        self.bytes_loaded = Counter("dcs_bytes_loaded_total")
        self.send_operations = Counter("dcs_send_operations_total")
        self.recv_operations = Counter("dcs_recv_operations_total")
        self.save_operations = Counter("dcs_save_operations_total")
        self.load_operations = Counter("dcs_load_operations_total")
        self.errors = Counter("dcs_errors_total")

        # Gauges
        self.memory_buffer_usage = Gauge("dcs_memory_buffer_usage_bytes")
        self.active_connections = Gauge("dcs_active_connections")
        self.pending_operations = Gauge("dcs_pending_operations")

        # Start time
        self._start_time = time.time()

    def record_save(self, bytes_saved: int, duration: float) -> None:
        """Record a checkpoint save operation.

        Args:
            bytes_saved: Number of bytes saved to storage
            duration: Time taken for the save operation (seconds)
        """
        self.save_latency.observe(duration)
        self.bytes_saved.inc(bytes_saved)
        self.save_operations.inc()

    def record_load(self, bytes_loaded: int, duration: float) -> None:
        """Record a checkpoint load operation.

        Args:
            bytes_loaded: Number of bytes loaded from storage
            duration: Time taken for the load operation (seconds)
        """
        self.load_latency.observe(duration)
        self.bytes_loaded.inc(bytes_loaded)
        self.load_operations.inc()

    def record_send(self, bytes_sent: int, duration: float) -> None:
        """Record a tensor send operation.

        Args:
            bytes_sent: Number of bytes transmitted
            duration: Time taken for the send operation (seconds)
        """
        self.send_latency.observe(duration)
        self.bytes_sent.inc(bytes_sent)
        self.send_operations.inc()

    def record_recv(self, bytes_received: int, duration: float) -> None:
        """Record a tensor receive operation.

        Args:
            bytes_received: Number of bytes received
            duration: Time taken for the receive operation (seconds)
        """
        self.recv_latency.observe(duration)
        self.bytes_received.inc(bytes_received)
        self.recv_operations.inc()

    def record_error(self, error_type: str = "unknown") -> None:
        """Record an error event.

        Args:
            error_type: Error type description (for logging/analysis)
        """
        self.errors.inc()

    def set_memory_usage(self, bytes_used: int) -> None:
        """Set the current memory buffer usage.

        Args:
            bytes_used: Current buffer memory in use (bytes)
        """
        self.memory_buffer_usage.set(bytes_used)

    def set_active_connections(self, count: int) -> None:
        """Set the number of active P2P connections.

        Args:
            count: Number of active connections
        """
        self.active_connections.set(count)

    def get_all(self) -> Dict[str, Any]:
        """Get all collected metrics as a dictionary.

        Returns:
            Dict with structure:
            {
                "uptime_seconds": float,
                "latency": {
                    "save": {...stats...},
                    "load": {...stats...},
                    "send": {...stats...},
                    "recv": {...stats...}
                },
                "counters": {
                    "bytes_sent": int,
                    "bytes_received": int,
                    "bytes_saved": int,
                    "bytes_loaded": int,
                    "send_operations": int,
                    "recv_operations": int,
                    "save_operations": int,
                    "load_operations": int,
                    "errors": int,
                },
                "gauges": {
                    "memory_buffer_usage": float,
                    "active_connections": float,
                    "pending_operations": float,
                }
            }
        """
        return {
            "uptime_seconds": time.time() - self._start_time,
            "latency": {
                "save": self.save_latency.get_stats(),
                "load": self.load_latency.get_stats(),
                "send": self.send_latency.get_stats(),
                "recv": self.recv_latency.get_stats(),
            },
            "counters": {
                "bytes_sent": self.bytes_sent.get(),
                "bytes_received": self.bytes_received.get(),
                "bytes_saved": self.bytes_saved.get(),
                "bytes_loaded": self.bytes_loaded.get(),
                "send_operations": self.send_operations.get(),
                "recv_operations": self.recv_operations.get(),
                "save_operations": self.save_operations.get(),
                "load_operations": self.load_operations.get(),
                "errors": self.errors.get(),
            },
            "gauges": {
                "memory_buffer_usage": self.memory_buffer_usage.get(),
                "active_connections": self.active_connections.get(),
                "pending_operations": self.pending_operations.get(),
            },
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Converts collected metrics to Prometheus-compatible format for scraping
        by Prometheus servers or other monitoring tools.

        Returns:
            str: Metrics in Prometheus text exposition format

        Example output:
            # HELP dcs_send_latency_seconds Histogram of dcs send latency seconds
            # TYPE dcs_send_latency_seconds histogram
            dcs_send_latency_seconds_bucket{le="0.001"} 42
            dcs_send_latency_seconds_bucket{le="0.01"} 156
            ...
            dcs_send_latency_seconds_sum 15.234
            dcs_send_latency_seconds_count 1023
        """
        lines = []

        # Helper to format histogram
        def format_histogram(name: str, histogram: Histogram) -> List[str]:
            result = []
            stats = histogram.get_stats()
            result.append(f"# HELP {name} Histogram of {name.replace('_', ' ')}")
            result.append(f"# TYPE {name} histogram")

            for bucket, count in sorted(stats.get("buckets", {}).items()):
                result.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            result.append(f'{name}_bucket{{le="+Inf"}} {stats["count"]}')
            result.append(f"{name}_sum {stats['sum']}")
            result.append(f"{name}_count {stats['count']}")
            return result

        # Helper to format counter
        def format_counter(name: str, counter: Counter) -> List[str]:
            return [
                f"# HELP {name} Counter of {name.replace('_', ' ')}",
                f"# TYPE {name} counter",
                f"{name} {counter.get()}",
            ]

        # Helper to format gauge
        def format_gauge(name: str, gauge: Gauge) -> List[str]:
            return [
                f"# HELP {name} Gauge of {name.replace('_', ' ')}",
                f"# TYPE {name} gauge",
                f"{name} {gauge.get()}",
            ]

        # Histograms
        lines.extend(format_histogram("dcs_save_latency_seconds", self.save_latency))
        lines.extend(format_histogram("dcs_load_latency_seconds", self.load_latency))
        lines.extend(format_histogram("dcs_send_latency_seconds", self.send_latency))
        lines.extend(format_histogram("dcs_recv_latency_seconds", self.recv_latency))

        # Counters
        lines.extend(format_counter("dcs_bytes_sent_total", self.bytes_sent))
        lines.extend(format_counter("dcs_bytes_received_total", self.bytes_received))
        lines.extend(format_counter("dcs_bytes_saved_total", self.bytes_saved))
        lines.extend(format_counter("dcs_bytes_loaded_total", self.bytes_loaded))
        lines.extend(format_counter("dcs_send_operations_total", self.send_operations))
        lines.extend(format_counter("dcs_recv_operations_total", self.recv_operations))
        lines.extend(format_counter("dcs_save_operations_total", self.save_operations))
        lines.extend(format_counter("dcs_load_operations_total", self.load_operations))
        lines.extend(format_counter("dcs_errors_total", self.errors))

        # Gauges
        lines.extend(format_gauge("dcs_memory_buffer_usage_bytes", self.memory_buffer_usage))
        lines.extend(format_gauge("dcs_active_connections", self.active_connections))
        lines.extend(format_gauge("dcs_pending_operations", self.pending_operations))

        # Uptime
        lines.extend(
            [
                "# HELP dcs_uptime_seconds Time since collector started",
                "# TYPE dcs_uptime_seconds gauge",
                f"dcs_uptime_seconds {time.time() - self._start_time}",
            ]
        )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.save_latency.reset()
        self.load_latency.reset()
        self.send_latency.reset()
        self.recv_latency.reset()
        self.bytes_sent.reset()
        self.bytes_received.reset()
        self.bytes_saved.reset()
        self.bytes_loaded.reset()
        self.send_operations.reset()
        self.recv_operations.reset()
        self.save_operations.reset()
        self.load_operations.reset()
        self.errors.reset()
        self._start_time = time.time()


# Global metrics instance (optional)
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector singleton.

    Creates a new instance on first call and reuses it for subsequent calls.
    Useful for accessing metrics from anywhere in the codebase without
    passing the collector instance around.

    Returns:
        MetricsCollector: The global metrics collector instance

    Example:
        metrics = get_metrics()
        metrics.record_send(bytes_sent=1024, duration=0.01)
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics collector.

    Clears all metrics and resets the start time. Useful for resetting state
    between test runs or benchmark iterations.
    """
    global _global_metrics
    if _global_metrics:
        _global_metrics.reset()
