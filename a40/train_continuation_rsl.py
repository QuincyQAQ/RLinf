"""Original PPO with weight-only initialization; no hanging Kit shutdown."""
import importlib.util
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmark/scripts'))
spec = importlib.util.spec_from_file_location('continuation_original_train', ROOT/'benchmark/scripts/train.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original = m.OnPolicyRunner.load
def load(self, path, *args, **kwargs):
    result = original(self, path, load_optimizer=False)
    self.current_learning_iteration = 0
    return result
m.OnPolicyRunner.load = load
if '--task' in sys.argv and 'B1' in sys.argv[sys.argv.index('--task')+1] and '--distribution_safety_patch' in sys.argv:
    from train_legacy_distribution import install
    install()
if os.environ.get('GO2_B1_NUMERIC_GUARD')=='1':
    from b1_numerical_boundary import install_gym_hook
    install_gym_hook()
code = 0
try:
    m.main()
except BaseException:
    traceback.print_exc()
    code = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
