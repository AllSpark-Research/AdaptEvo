# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Configuration classes for the Distributed Checkpoint Service."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BackendType(str, Enum):
    """Communication backend types for distributed tensor transmission.

    Attributes:
        GLOO: CPU-based collective communication (MPI-like, async)
        NCCL: GPU-optimized communication (NVIDIA Collective Communications Library)
        TCP: TCP socket-based communication for cross-cluster or long-distance transfers
    """

    GLOO = "gloo"
    NCCL = "nccl"
    TCP = "tcp"
    HCCL = "hccl"


class RoleInfo(BaseModel):
    """Information about a registered role/node in the DCS.

    A role represents a logical group of processes (e.g., "actor", "rollout", "trainer").
    Each role can have multiple ranks (processes).

    Attributes:
        role_name: Logical role name (e.g., "actor", "rollout", "trainer")
        rank: Rank (process ID) within the role. None if auto-assigned by coordinator
        world_size: Total number of processes in this role. None if not yet determined
        ip: IPv4 address of the node hosting this process
        port: Communication port for P2P connections
        device_id: GPU device ID if using GPU communication
        metadata: Additional metadata (e.g., tensor parallelism size, pipeline parallelism size)
    """

    role_name: str
    rank: int | None = None
    world_size: int | None = None
    ip: str | None = None
    port: int | None = None
    device_id: int | None = None
    metadata: Dict[str, Any] | None = None

    @property
    def node_id(self) -> str:
        """Generate a unique node identifier.

        Returns:
            str: Format "{role_name}_{rank}"
        """
        return f"{self.role_name}_{self.rank}"

    @property
    def address(self) -> str:
        """Generate full address string for network communication.

        Returns:
            str: Format "{ip}:{port}"
        """
        return f"{self.ip}:{self.port}"


class DCSConfig(BaseModel):
    """Configuration for the Distributed Checkpoint Service.

    This model defines all tunable parameters for DCS including:
    - Coordinator endpoints
    - Communication backend properties
    - Heartbeat and fault tolerance policies
    - Performance tuning parameters

    All settings have sensible defaults and can be overridden per deployment.
    """

    # Coordinator settings
    coordinator_host: str = Field(default="0.0.0.0", description="Coordinator bind host - IP address to listen on")
    coordinator_port: int = Field(default=8000, description="Coordinator HTTP port for REST API")

    # Backend settings
    backend_type: BackendType = Field(
        default=BackendType.GLOO, description="Default communication backend (GLOO, NCCL, or TCP)"
    )

    # Heartbeat settings
    heartbeat_interval_seconds: float = Field(
        default=5.0, gt=0, description="Interval between consecutive heartbeats from nodes"
    )
    heartbeat_timeout_seconds: float = Field(
        default=30.0, gt=0, description="Timeout to declare a node dead if no heartbeat received"
    )

    # Communication settings
    comm_base_port: int = Field(
        default=20000, description="Base port for P2P communication (ports auto-assigned from this base)"
    )
    tcp_nodelay: bool = Field(default=True, description="Disable Nagle's algorithm for TCP (reduces latency)")
    tcp_buffer_size: int = Field(default=65536, description="TCP send/receive buffer size in bytes (64 KB default)")

    # Storage settings
    checkpoint_dir: str = Field(
        default="/tmp/dcs_checkpoints", description="Directory path for storing checkpoint files"
    )
    async_io: bool = Field(default=True, description="Enable async I/O for checkpoint save/load operations")

    # Performance settings
    tensor_fusion_threshold: int = Field(
        default=1024 * 1024,  # 1MB
        description="Minimum total tensor size to trigger fusion optimization (bytes)",
    )
    pinned_memory: bool = Field(
        default=True, description="Use CUDA pinned memory for efficient device-to-host transfers"
    )

    # Fault tolerance
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts for failed operations")
    retry_delay_seconds: float = Field(
        default=1.0, gt=0, description="Delay (exponential backoff) between retry attempts"
    )

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus-compatible metrics collection")
    metrics_port: int = Field(default=9090, description="Port for Prometheus metrics HTTP endpoint")


class TopologyConfig(BaseModel):
    """Configuration for topology mapping between roles.

    Defines how logical roles connect to each other for tensor exchange.
    Used by the coordinator to establish communication groups between roles.

    Attributes:
        role_mappings: Dict mapping source roles to destination roles
                      Example: {'actor': 'rollout'} means actor:rank connects to rollout:rank
    """

    role_mappings: Dict[str, str] = Field(
        default_factory=dict, description="Role mappings dict, e.g., {'actor': 'rollout'} for role-to-role connections"
    )

    def get_peer_role(self, role: str) -> Optional[str]:
        """Get the peer role for a given role.

        Args:
            role: Source role name

        Returns:
            str: Peer role name, or None if no mapping exists
        """
        return self.role_mappings.get(role)
