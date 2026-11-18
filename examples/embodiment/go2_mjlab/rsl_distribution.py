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

"""The reference B2 Gaussian bounds without changing upstream PPO."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class BoundedGaussianDistribution(GaussianDistribution):
    """Project scalar standard deviations before constructing the Gaussian."""

    def update(self, mlp_output: torch.Tensor) -> None:
        """Apply the successful B2 recipe's finite and standard-deviation bounds."""
        with torch.no_grad():
            self.std_param.nan_to_num_(nan=1.0, posinf=5.0, neginf=1.0e-6)
            self.std_param.clamp_(min=1.0e-6, max=5.0)
        super().update(
            torch.nan_to_num(mlp_output, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        )
