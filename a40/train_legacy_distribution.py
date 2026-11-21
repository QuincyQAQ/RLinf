"""Compatibility for the upstream requested safety patch on bundled RSL API.
Original benchmark files remain unchanged. This repaired run is explicitly tagged.
"""
import runpy
import sys
from pathlib import Path


def safe_std(value):
    import torch
    return torch.nan_to_num(value, nan=1.0, posinf=5.0, neginf=1e-6).clamp(1e-6, 5.0)


def install():
    import torch
    from torch.distributions import Normal
    from rsl_rl.modules.actor_critic import ActorCritic
    original = ActorCritic._update_distribution

    def update(self, obs):
        if not self.state_dependent_std and self.noise_std_type == 'scalar':
            with torch.no_grad():
                self.std.copy_(safe_std(self.std))
        original(self, obs)
        mean = torch.nan_to_num(self.distribution.loc, nan=0.0, posinf=1e6, neginf=-1e6)
        self.distribution = Normal(mean, safe_std(self.distribution.scale))

    ActorCritic._update_distribution = update
    print('[RECOVERY] Applied requested distribution safety to bundled ActorCritic API', flush=True)


if __name__ == '__main__':
    if '--distribution_safety_patch' not in sys.argv:
        raise SystemExit('Compatibility entry requires explicit original safety flag')
    install()
    target = Path(__file__).resolve().parents[1] / 'benchmark/scripts/train.py'
    sys.argv[0] = str(target)
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name='__main__')
