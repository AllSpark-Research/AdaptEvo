# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from argparse import Namespace
from typing import Any, Optional

import ray
import transfer_queue as tq
from ray import serve

from relax.components.base import Base
from relax.distributed.ray.placement_group import allocate_train_group


@serve.deployment
class Critic(Base):
    """Critic service for training the value model."""

    def __init__(
        self, healthy: Any, pgs: Optional[Any], num_gpus: int, config: Namespace, role: str, runtime_env: dict = None
    ) -> None:
        super().__init__()

        self.config = config
        self._lock = threading.RLock()
        self.healthy = healthy
        self.role = role

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()

        self.critic_model = allocate_train_group(args=config, num_gpus=num_gpus, pg=pgs, runtime_env=runtime_env)

        ray.get(self.critic_model.async_init(config, role=self.role, with_ref=False))
        self.step = 0

    async def run(self) -> None:
        try:
            self.train()
        except Exception as e:
            error_msg = f"Critic training failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("critic", error_msg)
            if not getattr(self.config, "use_health_check", False):
                raise

    def train(self) -> None:
        while self.step < self.config.num_rollout:
            self.critic_model.async_train(self.step)
            if self.config.save_interval is not None and (
                (self.step + 1) % self.config.save_interval == 0 or (self.step + 1 == self.config.num_rollout)
            ):
                self.critic_model.save_model(self.step)

            try:
                self.healthy.update_heartbeat.remote("critic", self.step + 1)
            except Exception:
                pass

            self.step += 1
