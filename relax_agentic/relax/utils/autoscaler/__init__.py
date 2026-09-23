# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Autoscaler module for dynamic Rollout engine scaling.

This module provides automatic scaling of SGLang inference engines based on
real-time metrics collection and configurable scaling policies.

Key Components:
    - AutoscalerConfig: Configuration dataclass for autoscaler behavior
    - MetricsCollector: Collects and aggregates metrics from engines
    - ScalingDecisionEngine: Evaluates metrics and decides scaling actions
    - AutoscalerService: Ray Serve deployment for the autoscaler

Usage:
    The autoscaler is enabled by providing a YAML config file via the
    --autoscaler-config command-line argument:

    python relax/entrypoints/train.py --autoscaler-config relax/utils/autoscaler/autoscaler.yaml

    If --autoscaler-config is not provided, the autoscaler is disabled.

Example:
    >>> from relax.utils.autoscaler import AutoscalerConfig
    >>> config = AutoscalerConfig.from_yaml("relax/utils/autoscaler/autoscaler.yaml")
"""

from pathlib import Path

from relax.utils.autoscaler.autoscaler_service import (
    AutoscalerService,
    ScaleHistoryItem,
    ScaleHistoryResponse,
)
from relax.utils.autoscaler.config import AutoscalerConfig, ScaleInPolicy, ScaleOutPolicy
from relax.utils.autoscaler.metrics_collector import EngineMetrics, MetricsCollector
from relax.utils.autoscaler.scaling_decision import (
    ScalingAction,
    ScalingDecision,
    ScalingDecisionEngine,
)


DEFAULT_CONFIG_PATH = Path(__file__).parent / "autoscaler.yaml"

__all__ = [
    "AutoscalerConfig",
    "AutoscalerService",
    "DEFAULT_CONFIG_PATH",
    "EngineMetrics",
    "MetricsCollector",
    "ScaleHistoryItem",
    "ScaleHistoryResponse",
    "ScaleInPolicy",
    "ScaleOutPolicy",
    "ScalingAction",
    "ScalingDecision",
    "ScalingDecisionEngine",
]
