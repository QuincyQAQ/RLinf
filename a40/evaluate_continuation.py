"""Seeded original evaluation plus passage metrics; bounded headless exit."""
import os
import evaluate_dual

original_augment = evaluate_dual.augment_evaluator

def augment(module):
    original_make = module.gym.make
    def make(task, *args, **kwargs):
        seed = int(os.environ.get('GO2_EVAL_SEED', '12345'))
        kwargs['cfg'].seed = seed
        if 'B1' in str(task) and os.environ.get('GO2_B1_NUMERIC_GUARD')=='1':
            from b1_numerical_boundary import configure
            configure(kwargs['cfg'])
        module.torch.manual_seed(seed)
        return original_make(task, *args, **kwargs)
    module.gym.make = make
    original_augment(module)

evaluate_dual.augment_evaluator = augment
if __name__ == '__main__':
    evaluate_dual.main()
