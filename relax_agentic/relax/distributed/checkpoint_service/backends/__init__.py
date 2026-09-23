# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Communication backends package."""

from relax.distributed.checkpoint_service.backends.base import CommBackend
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


__all__ = [
    "CommBackend",
    "DeviceDirectBackend",
]
