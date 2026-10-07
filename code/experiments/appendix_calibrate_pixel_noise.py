"""Calibrate the noise intensities of the camera environment (chapter 6).

Run with::

    python -m experiments.appendix_calibrate_pixel_noise
"""

import json

import numpy as np

import settings
from experiments.appendix_calibrate_noise import calibrate
from experiments.ch6_noise import pixel_policy
from experiments.noise_builder import build_noise
from experiments.registry import _pixel_protocol, _pixel_setup
from experiments.store import experiment_root, in_shard
from seeding import sub_seed

EXPERIMENT = "appendix_calibrate_pixel_noise"

REFERENCE = "oracle_coarse"

# Don't calibrate measurement
CALIBRATED_CHANNELS = tuple(c for c in settings.NOISE_CHANNELS if c != "measurement")


class _Reference:
    solver = REFERENCE


def mean_cost(channel=None, std=None) -> float:
    """Test cost of the oracle pooled over all seeds."""
    from reinforcement_learning.pixel_env import build_pixel_env

    K, _, horizon, _ = _pixel_setup(None)
    noise = None if channel is None else build_noise(channel, "calibration", {channel: std})

    costs = []
    with np.errstate(all="ignore"):
        for seed in settings.SEEDS:
            env = build_pixel_env(sub_seed(seed, "pixel-score", REFERENCE), horizon=horizon)
            env.lqr.setup_noise(noise)
            evaluate_test, _ = _pixel_protocol(env, pixel_policy(_Reference, env, K), horizon)
            costs.append(evaluate_test().summary["test_cost_mean"])
    return float(np.mean(costs))


def main():
    clean = mean_cost()

    for index, channel in enumerate(CALIBRATED_CHANNELS):
        if not in_shard(index):
            continue

        result = {"none": {"std": 0.0, "ratio": 1.0}}
        for level, target in settings.NOISE_COST_TARGETS.items():
            std = calibrate(lambda std: mean_cost(channel, std), target, clean)
            ratio = mean_cost(channel, std) / clean
            result[level] = {"std": std, "target": target, "ratio": ratio}
            print(f"{channel:>12} {level:>10}: std={std:<10g} ratio={ratio:.3f} (target {target})")

        path = experiment_root(EXPERIMENT) / f"calibration_{channel}.json"
        path.write_text(json.dumps({
            "reference": REFERENCE,
            "disturbance_type": settings.DISTURBANCE_TYPE,
            "parametric_time_invariant": settings.PARAMETRIC_TIME_INVARIANT,
            "clean_cost": clean,
            "levels": {channel: result},
        }, indent=2))


if __name__ == "__main__":
    main()
