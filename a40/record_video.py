#!/usr/bin/env python3
"""Record a trained reference policy without training or changing benchmark files."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['A2', 'B1', 'B2'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='RSL checkpoint or RLinf inference export, not native full_weights.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--execute', action='store_true', help='Otherwise print a CPU-only plan')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or min(args.width, args.height) <= 0:
        parser.error('Duration and resolution must be positive and finite')
    if args.width % 2 or args.height % 2:
        parser.error('H.264 resolution must use even width and height')
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if output.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4')
    plan = dict(task=args.task, checkpoint=str(checkpoint), output=str(output),
                seconds=args.seconds, resolution=[args.width, args.height], num_envs=1,
                training=False, quantitative_evaluation=False)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('GPU recording requires an allocated Slurm compute job')
    if not checkpoint.is_file():
        parser.error('Checkpoint does not exist')
    partial = output.with_name(output.stem + '.partial.mp4')
    report = output.with_suffix('.json')
    if any(p.exists() for p in (output, partial, report)):
        parser.error('Output already exists; select a new filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'config/reference.json').read_text())
    task_id = manifest['tasks'][args.task]['eval_gym_id']
    sys.path.insert(0, str(ROOT / 'benchmark/scripts'))
    sys.argv = [str(ROOT / 'benchmark/scripts/evaluate.py'), '--task', task_id,
                '--checkpoint', str(checkpoint), '--num_envs', '1', '--headless', '--enable_cameras']
    spec = importlib.util.spec_from_file_location('video_reference_evaluator', ROOT / 'benchmark/scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = None
    try:
        import imageio.v2 as imageio

        @module.hydra_task_config(task_id, 'rsl_rl_cfg_entry_point')
        def record(env_cfg, agent_cfg):
            nonlocal env
            device = module.args_cli.device or 'cuda:0'
            env_cfg.scene.num_envs = 1
            env_cfg.sim.device = device
            env_cfg.viewer.resolution = (args.width, args.height)
            env_cfg.viewer.origin_type = 'asset_root'
            env_cfg.viewer.asset_name = 'robot'
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.eye = (3.0, 3.0, 2.0)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
            env = module.gym.make(task_id, cfg=env_cfg, render_mode='rgb_array')
            wrapped = module.RslRlVecEnvWrapper(env)
            module._align_agent_cfg_to_checkpoint(agent_cfg, str(checkpoint))
            agent_cfg = module.handle_deprecated_rsl_rl_cfg(agent_cfg, module.metadata.version('rsl-rl-lib'))
            module._align_agent_cfg_to_checkpoint(agent_cfg, str(checkpoint))
            runner = module.OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
            module._load_runner_checkpoint_compat(runner, str(checkpoint))
            policy = runner.get_inference_policy(device=device)
            obs = wrapped.get_observations()
            step_dt = env.unwrapped.step_dt
            steps = math.ceil(args.seconds / step_dt)
            with imageio.get_writer(str(partial), fps=1.0 / step_dt, codec='libx264', macro_block_size=1) as writer:
                with module.torch.inference_mode():
                    for _ in range(steps):
                        obs, _, _, _ = wrapped.step(policy(obs))
                        frame = env.render()
                        if frame is None or frame.ndim != 3:
                            raise RuntimeError('Renderer did not return an RGB frame')
                        writer.append_data(frame)
            # Decode the completed file before declaring a usable video.
            with imageio.get_reader(str(partial)) as reader:
                count = reader.count_frames()
                first = reader.get_data(0)
                last = reader.get_data(count - 1)
            if count != steps or first.shape[:2] != (args.height, args.width) or last.size == 0:
                raise RuntimeError('Encoded video failed frame count or resolution validation')
            partial.rename(output)
            report.write_text(json.dumps(dict(plan, status='verified', frames=count,
                fps=1.0 / step_dt, simulated_seconds=steps * step_dt,
                camera='robot-follow', task_id=task_id), indent=2))
            print('VIDEO_VERIFIED:', output, flush=True)

        record()
    finally:
        if env is not None:
            env.close()
        module.simulation_app.close()


if __name__ == '__main__':
    main()
