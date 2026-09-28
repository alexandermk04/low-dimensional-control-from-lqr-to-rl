"""Calibrate the noise intensities by the cost increase they cause.

Run with::

    python -m experiments.appendix_calibrate_noise
"""

import json

import numpy as np

import settings
from evaluation.metrics import LinearPolicy
from experiments.evaluate import evaluate_policy
from experiments.noise_builder import build_noise
from experiments.problem import make_problem
from experiments.registry import optimal_controller_gain
from experiments.store import experiment_root

EXPERIMENT = "appendix_calibrate_noise"

# Relative tolerance on the reached cost ratio
TOLERANCE = 0.01

# Log-space search range for the intensities
SEARCH_RANGE = (1e-6, 1e3)


def mean_cost(problems, channel=None, std=None) -> float:
    """Test cost of the optimal controller pooled over all seeds."""
    noise = None if channel is None else build_noise(channel, "calibration", {channel: std})
    with np.errstate(all="ignore"):
        costs = [
            evaluate_policy(problem, LinearPolicy(optimal_controller_gain(problem)),
                            noise_settings=noise).returns
            for problem in problems
        ]
    return float(np.mean(costs))


def calibrate(problems, channel: str, target: float, clean: float) -> float:
    """Bisect the intensity in log space until the cost ratio hits ``target``.

    Diverging costs count as a too large intensity.
    """
    low, high = np.log10(SEARCH_RANGE)
    for _ in range(60):
        middle = 0.5 * (low + high)
        ratio = mean_cost(problems, channel, 10 ** middle) / clean
        if np.isfinite(ratio) and abs(ratio / target - 1) < TOLERANCE:
            break
        if np.isfinite(ratio) and ratio < target:
            low = middle
        else:
            high = middle
    return float(f"{10 ** middle:.3g}")


def main():
    base = settings.INFLATION_SIZES["base"]
    problems = [make_problem(seed, base) for seed in settings.SEEDS]

    clean = mean_cost(problems)

    result = {}
    for channel in settings.NOISE_CHANNELS:
        result[channel] = {"none": {"std": 0.0, "ratio": 1.0}}
        for level, target in settings.NOISE_COST_TARGETS.items():
            std = calibrate(problems, channel, target, clean)
            ratio = mean_cost(problems, channel, std) / clean
            result[channel][level] = {"std": std, "target": target, "ratio": ratio}
            print(f"{channel:>12} {level:>10}: std={std:<10g} ratio={ratio:.3f} (target {target})")

    path = experiment_root(EXPERIMENT) / "calibration.json"
    path.write_text(json.dumps({
        "inflation": list(base),
        "disturbance_type": settings.DISTURBANCE_TYPE,
        "parametric_time_invariant": settings.PARAMETRIC_TIME_INVARIANT,
        "clean_cost": clean,
        "levels": result,
    }, indent=2))

    print("\nNOISE_STD = {")
    for channel, levels in result.items():
        print(f'    "{channel}": {{')
        for level, entry in levels.items():
            print(f'        "{level}": {entry["std"]},')
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
