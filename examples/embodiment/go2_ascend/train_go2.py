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

"""Launch the Go2 compatibility policy/worker through RLinf's embodied runner."""

from pathlib import Path

import go2_rsl_extension
import hydra
import torch.multiprocessing as mp
from go2_rsl_extension import Go2EnvWorker, RslGo2Actor, SeededGo2Rollout
from hydra.utils import to_absolute_path

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement


def progress_flags(current, target, save_interval, eval_interval):
    final = current >= target
    save = save_interval > 0 and (current % save_interval == 0 or final)
    evaluate = eval_interval > 0 and (current % eval_interval == 0 or final)
    return save, evaluate


class Go2Runner(EmbodiedRunner):
    """Keep the previous experiment's independent save and evaluation intervals."""

    def _maybe_eval_and_checkpoint(self, step):
        save, evaluate = progress_flags(
            self.global_step,
            self.max_steps,
            self.cfg.runner.save_interval,
            self.cfg.runner.val_check_interval,
        )
        metrics = {}
        if evaluate:
            with self.timer("eval"):
                self.update_rollout_weights()
                metrics = {f"eval/{k}": v for k, v in self.evaluate().items()}
                self.metric_logger.log(data=metrics, step=step)
        if save:
            self._save_checkpoint()
        return metrics


@hydra.main(version_base="1.1", config_path=".", config_name="single_smoke")
def main(cfg):
    go2_rsl_extension.register()
    cfg.runner.logger.log_path = to_absolute_path(cfg.runner.logger.log_path)
    if cfg.runner.resume_dir:
        cfg.runner.resume_dir = to_absolute_path(cfg.runner.resume_dir)
    if cfg.actor.model.initial_checkpoint:
        cfg.actor.model.initial_checkpoint = to_absolute_path(
            cfg.actor.model.initial_checkpoint
        )
    Path(cfg.runner.logger.log_path).mkdir(parents=True, exist_ok=True)
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = RslGo2Actor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    rollout = SeededGo2Rollout.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = Go2EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = Go2Runner(cfg=cfg, actor=actor, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
