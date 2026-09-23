# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Abstract base class for communication backends.

Provides a unified interface for different communication mechanisms:
- DeviceDirect: NCCL/GLOO for GPU-to-GPU or CPU distributed communication
- CpuOffload: TCP-based for cross-cluster or long-distance communication
"""

import asyncio
import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from relax.distributed.checkpoint_service.config import BackendType, RoleInfo


logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class SendRequest:
    """A request to send tensors to a destination node.

    Represents a point-to-point send operation with metadata and options.

    Attributes:
        tensor_dict: Dictionary of {tensor_name: tensor} where tensor is torch.Tensor or ndarray
        dst_rank: Destination node rank
        group_name: Optional process group name for group communication
        async_op: If True, return immediately with a CommHandle; if False, block until complete
        metadata: Optional additional metadata to include (e.g., {"checkpoint_step": 100})
    """

    tensor_dict: Dict[str, Any]  # torch.Tensor or ndarray
    dst_rank: int
    group_name: Optional[str] = None
    async_op: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecvRequest:
    """A request to receive tensors from a source node.

    Represents a point-to-point receive operation with optional shape hints.

    Attributes:
        src_rank: Source node rank
        tensor_names: Optional list of expected tensor names (for validation/routing)
        group_name: Optional process group name for group communication
        metadata: Optional metadata associated with the receive (populated by receive operation)
    """

    src_rank: int
    tensor_names: Optional[List[str]] = None
    group_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommHandle:
    """Handle for async communication operations.

    Allows checking completion status and waiting for async operations to finish.
    Provides both blocking and async interfaces.

    Attributes:
        request_id: Unique identifier for the operation
        is_complete: Whether the operation has completed
        result: Operation result (populated when complete)
        error: Exception if operation failed, None otherwise
    """

    request_id: str
    is_complete: bool = False
    result: Optional[Any] = None
    error: Optional[Exception] = None

    def wait(self) -> Any:
        """Wait for the operation to complete (blocking).

        Returns:
            The operation result

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    async def wait_async(self) -> Any:
        """Wait for the operation to complete (async).

        Returns:
            The operation result

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError


class CommBackend(ABC):
    """Abstract base class for communication backends.

    Defines a unified interface for different communication mechanisms:
    - DeviceDirectBackend: NCCL (GPU) or GLOO (CPU) collective communication
    - CpuOffloadTcpBackend: TCP-based for cross-cluster communication

    Subclasses must implement:
    - send(): Send tensors to a destination rank
    - recv(): Receive tensors from a source rank
    - init_process_group(): Initialize distributed communication
    - close()/close_async(): Clean up resources

    Example:
        backend = DeviceDirectBackend(BackendType.NCCL, role_info)
        backend.init_process_group()

        # Send
        handle = backend.send({"weight": tensor}, dst=1, async_op=True)
        handle.wait()

        # Receive
        tensors = backend.recv(src=0)
    """

    def __init__(
        self,
        backend_type: BackendType = BackendType.GLOO,
        role_info: Optional[RoleInfo] = None,
    ):
        """Initialize the communication backend.

        Args:
            backend_type: Type of backend (GLOO, NCCL, TCP)
            role_info: Information about the current node/rank
        """
        self.backend_type = backend_type
        self.role_info = role_info
        self._initialized = False
        self._groups: Dict[str, Any] = {}  # group_name -> ProcessGroup or connection pool
        self._peer_info: Dict[int, RoleInfo] = {}  # rank -> RoleInfo
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Check if the backend is initialized."""
        return self._initialized

    @property
    def rank(self) -> int:
        """Get the current rank of this process.

        Returns:
            int: Rank within the communication group

        Raises:
            RuntimeError: If role_info not set
        """
        if self.role_info is None:
            raise RuntimeError("Role info not set")
        return self.role_info.rank

    def register_peer(self, peer_info: RoleInfo) -> None:
        """Register a peer node for communication.

        Args:
            peer_info: Information about the peer node
        """
        self._peer_info[peer_info.rank] = peer_info
        logger.debug(f"Registered peer: {peer_info.node_id} at {peer_info.address}")

    def get_peer(self, rank: int) -> Optional[RoleInfo]:
        """Get peer info by rank.

        Args:
            rank: Rank of the peer

        Returns:
            RoleInfo if found, None otherwise
        """
        return self._peer_info.get(rank)

    def get_all_peers(self) -> Dict[int, RoleInfo]:
        """Get all registered peers.

        Returns:
            Dict of {rank: RoleInfo} for all registered peers
        """
        return self._peer_info.copy()

    def init_process_group(self) -> None:
        """Initialize the process group for distributed communication.

        Should be called after all peers are registered. Default implementation
        does nothing; subclasses may override.
        """
        pass

    def broadcast(
        self,
        tensor_dict: Dict[str, Any],
        src: int,
        group: Optional[str] = None,
        async_op: bool = True,
    ) -> Optional[CommHandle]:
        """Broadcast tensors from source to all ranks.

        Args:
            tensor_dict: Tensors to broadcast (only meaningful at src rank)
            src: Source rank
            group: Optional process group name
            async_op: If True, return immediately with a handle

        Returns:
            CommHandle if async_op=True, None if async_op=False

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Broadcast not implemented for this backend")

    async def broadcast_async(
        self,
        tensor_dict: Dict[str, Any],
        src: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of broadcast.

        Args:
            tensor_dict: Tensors to broadcast
            src: Source rank
            group: Optional process group name

        Returns:
            Dict of {tensor_name: tensor}

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Broadcast not implemented for this backend")

    def create_group(self, name: str, ranks: List[int]) -> Any:
        """Create a new communication group.

        Args:
            name: Group name for reference
            ranks: List of ranks to include in the group

        Returns:
            Group handle (backend-specific)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("create_group not implemented for this backend")

    def destroy_group(self, name: str) -> None:
        """Destroy a communication group.

        Args:
            name: Name of the group to destroy
        """
        if name in self._groups:
            del self._groups[name]


class TensorFusion:
    """Utility for fusing multiple small tensors into a larger buffer.

    Tensor fusion reduces communication overhead when transmitting many small tensors
    by concatenating them into a single large buffer, reducing protocol overhead.

    Example:
        fusion = TensorFusion(threshold_bytes=1024*1024)  # 1MB threshold

        if fusion.should_fuse({"w1": t1, "w2": t2}):
            fused, metadata = fusion.fuse({"w1": t1, "w2": t2})
            # Send fused buffer...
            unfused = fusion.unfuse(fused, metadata)
    """

    def __init__(self, threshold_bytes: int = 1024 * 1024):
        """Initialize tensor fusion.

        Args:
            threshold_bytes: Minimum total tensor size to trigger fusion (default: 1MB)
                            Fusion is only applied if total size >= threshold
        """
        self.threshold = threshold_bytes

    def should_fuse(self, tensor_dict: Dict[str, Any]) -> bool:
        """Check if tensors should be fused.

        Fusion is applied if:
        1. PyTorch is available
        2. Total tensor size >= threshold
        3. More than one tensor

        Args:
            tensor_dict: Dictionary of tensors to evaluate

        Returns:
            bool: True if fusion should be applied
        """
        if not TORCH_AVAILABLE:
            return False

        total_size = 0
        for tensor in tensor_dict.values():
            if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                total_size += tensor.numel() * tensor.element_size()

        return total_size >= self.threshold and len(tensor_dict) > 1

    def fuse(self, tensor_dict: Dict[str, Any]) -> tuple:
        """Fuse tensors into a single buffer.

        Flattens all tensors and concatenates them into a single large tensor.
        Returns metadata needed to reconstruct the original tensors.

        Args:
            tensor_dict: Dictionary of {tensor_name: tensor}

        Returns:
            Tuple of (fused_tensor, metadata_list) where:
                - fused_tensor: Single large torch.Tensor
                - metadata_list: List of dicts with original shape/dtype/name for each tensor

        Raises:
            RuntimeError: If PyTorch not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for tensor fusion")

        # Flatten all tensors
        flat_tensors = []
        metadata = []

        for name, tensor in tensor_dict.items():
            flat = tensor.flatten()
            metadata.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
            )
            flat_tensors.append(flat)

        # Concatenate
        fused = torch.cat(flat_tensors)
        return fused, metadata

    def unfuse(self, fused_buffer: Any, metadata: List[Dict]) -> Dict[str, Any]:
        """Unfuse a buffer back into individual tensors.

        Reconstructs original tensors from fused buffer using metadata.

        Args:
            fused_buffer: The fused tensor from fuse()
            metadata: Metadata list from fuse()

        Returns:
            Dict of {tensor_name: tensor} with original shapes and dtypes

        Raises:
            RuntimeError: If PyTorch not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for tensor fusion")

        result = {}
        offset = 0

        for item in metadata:
            numel = item["numel"]
            flat = fused_buffer[offset : offset + numel]
            tensor = flat.reshape(item["shape"])
            result[item["name"]] = tensor
            offset += numel

        return result
