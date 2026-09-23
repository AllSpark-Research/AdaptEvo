# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
Distributed Checkpoint Service (DCS) - A high-performance checkpoint engine.

This package provides a distributed checkpoint engine with:
- Control plane / Data plane separation
- Dynamic role-aware networking
- Dual communication backends (DeviceDirect + CpuOffload)
- Elastic scaling and resharding support
- Production-grade fault tolerance
"""

from relax.distributed.checkpoint_service.backends import CommBackend, DeviceDirectBackend
from relax.distributed.checkpoint_service.client import CheckpointEngineClient
from relax.distributed.checkpoint_service.config import BackendType, DCSConfig, RoleInfo
from relax.distributed.checkpoint_service.coordinator import DCSCoordinator
from relax.distributed.checkpoint_service.metrics import MetricsCollector


__version__ = "0.1.0"

__all__ = [
    "DCSConfig",
    "RoleInfo",
    "BackendType",
    "CommBackend",
    "DeviceDirectBackend",
    "DCSCoordinator",
    "CheckpointEngineClient",
    "MetricsCollector",
]
