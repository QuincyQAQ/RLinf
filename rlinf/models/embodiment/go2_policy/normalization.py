# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpointed observation moments shared by PPO actor ranks."""

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize observations with population moments and epsilon 0.01."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, size))
        self.register_buffer("variance", torch.ones(1, size))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Apply frozen moments during inference and gradient computation."""
        return (observations - self.mean) / (self.variance.sqrt() + 0.01)

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        """Merge a complete rollout, including other distributed actor ranks."""
        values = observations.reshape(-1, self.mean.shape[-1]).to(
            device=self.mean.device, dtype=torch.float32
        )
        moments = torch.cat(
            (values.sum(0), values.square().sum(0), values.new_tensor([len(values)]))
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(moments)
        total, squares, count = torch.tensor_split(
            moments, (values.shape[-1], 2 * values.shape[-1])
        )
        batch_mean = total / count
        batch_variance = (squares / count - batch_mean.square()).clamp_min(0)
        combined_count = self.count + count[0].to(torch.int64)
        rate = count / combined_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * rate
        self.variance.add_(
            rate * (batch_variance - self.variance + delta * (batch_mean - new_mean))
        )
        self.variance.clamp_min_(0)
        self.mean.copy_(new_mean)
        self.count.copy_(combined_count)
