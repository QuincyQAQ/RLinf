"""One new B1 evaluation seed for all three frozen, completed checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from reference import ROOT, SHORT, evaluation_argv, verify_source
from b2_continuation import run, validate, write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    count = sum(row['num_episodes'] for row in rows)
    successes = sum(round(row['num_episodes'] * row['success_rate']) for row in rows)
    return dict(
        episodes=count, successes=successes, tracking_rate=successes / count,
        **{key: sum(row[key] * row['num_episodes'] for row in rows) / count
           for key in ['mean_lin_vel_error', 'mean_ang_vel_error',
                       'mean_alive_time_s', 'mean_success_step_ratio', 'failure_rate']})


def evaluate(python, checkpoint, directory, seed, minimum):
    command = evaluation_argv('B1', 0, checkpoint, python, directory)
    command[1] = str(ROOT / 'a40/evaluate_continuation.py')
    command[command.index('--num_episodes') + 1] = str(minimum)
    command.append('--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry '
                   '--/persistent/isaac/asset_root/cloud=' + str(ROOT / 'assets/isaac_sim_5.1'))
    run(command, directory, 1800, GO2_EVAL_SEED=str(seed), GO2_B1_NUMERIC_GUARD='1')
    artifact = directory / (SHORT['B1'] + '_s0') / (SHORT['B1'] + '.json')
    data = json.loads(artifact.read_text())
    episodes, summary = data['episodes'], data['summary']
    assert len(episodes) >= minimum and len(episodes) == summary['num_episodes']
    assert all(row.get('episode_success') is not None for row in episodes)
    successes = sum(bool(row['episode_success']) for row in episodes)
    assert abs(successes / len(episodes) - summary['success_rate']) < 1e-10
    assert '[B1_ACTION_BOUND]' in (directory / 'console.log').read_text(errors='replace')
    write(directory / 'verified.json', dict(status='passed', seed=seed,
          artifact=str(artifact), successes=successes, summary=summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True)
    parser.add_argument('--seed', type=int, default=45678)
    parser.add_argument('--episodes', type=int, default=2048)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    assert args.episodes >= 2048
    verify_source()
    source = ROOT / 'outputs/fullppo_huber_B1_14032301'
    original = json.loads((source / 'result.json').read_text())
    assert original['status'] == 'verified'
    assert args.seed not in original['protocol']['holdout_seeds']
    models = {
        'baseline': original['protocol']['source'],
        'rlinf_huber': str(source / 'rlinf_huber/training/policy.pt'),
        'rsl_control': json.loads((source / 'rsl_control/training/validation.json').read_text())['checkpoint'],
    }
    hashes = {lane: digest(path) for lane, path in models.items()}
    proofs = {lane: validate(Path(path)) for lane, path in models.items()}
    protocol = dict(task='B1', training_seed=0, new_evaluation_seed=args.seed,
                    episodes_per_lane_minimum=args.episodes, num_envs=256,
                    all_lanes_same_seed=True, models=models,
                    original_result=str(source / 'result.json'),
                    numerical_repair='same action/history bound +/-100 in all lanes; original rewards and terminations',
                    selection='single prespecified seed; retain every outcome and original results; no winner-dependent reruns',
                    scope='evaluation-only; fixed checkpoints, no training or policy tuning')
    if args.cpu_only:
        import torch
        assert not torch.cuda.is_initialized()
        write(ROOT / 'validation/b1_confirmatory_cpu.json', dict(
            status='passed', cuda_initialized=False, protocol=protocol, checkpoints=proofs, sha256=hashes))
        print('B1_CONFIRMATORY_CPU_PASSED', flush=True)
        return
    assert os.environ.get('SLURM_JOB_ID'), 'Requires allocated GPU'
    root = ROOT / 'outputs' / ('b1_confirmatory_' + os.environ['SLURM_JOB_ID'])
    root.mkdir()
    write(root / 'protocol.json', protocol)
    write(root / 'checkpoint_gate.json', dict(status='passed', checkpoints=proofs, sha256=hashes))
    try:
        fresh = {}
        for lane, path in models.items():
            write(root / 'progress.json', dict(stage='evaluation', time=time.time(), lane=lane, seed=args.seed))
            fresh[lane] = evaluate(args.python, Path(path), root / lane, args.seed, args.episodes)
            write(root / 'partial_results.json', fresh)
        assert hashes == {lane: digest(path) for lane, path in models.items()}, 'Checkpoint changed'
        write(root / 'result.json', dict(
            status='verified', protocol=protocol, sha256=hashes, new_seed_results=fresh,
            new_seed_summary={lane: summarize([row]) for lane, row in fresh.items()},
            pooled_original_and_new={lane: summarize(list(original['results'][lane].values()) + [row])
                                     for lane, row in fresh.items()},
            caveat='one training seed; evaluation randomness near ceiling; no guaranteed RLinf ranking'))
        write(root / 'progress.json', dict(stage='completed', time=time.time()))
    except BaseException as exc:
        write(root / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        write(root / 'progress.json', dict(stage='failed', time=time.time(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
