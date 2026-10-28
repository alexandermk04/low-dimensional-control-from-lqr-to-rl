"""Calibrate the noise intensities by the cost increase they cause.

Run with::

    python -m experiments.appendix_calibrate_noise

Afterwards, the cost ratio of the optimal controller is measured per inflation
size. ``python -m experiments.appendix_calibrate_noise sizes`` only repeats this
step, with the stored intensities.
"""

import json
import sys

import numpy as np

import settings
from evaluation.metrics import LinearPolicy
from experiments.evaluate import evaluate_policy
from experiments.noise_builder import build_noise
from experiments.problem import make_problem
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
            evaluate_policy(problem, LinearPolicy(problem.K_star),
                            noise_settings=noise).returns
            for problem in problems
        ]
    return float(np.mean(costs))


def calibrate(cost, target: float, clean: float) -> float:
    """Bisect the intensity in log space until the cost ratio hits ``target``.

    ``cost(std)`` is the pooled test cost at that intensity. Diverging costs
    count as a too large intensity.
    """
    low, high = np.log10(SEARCH_RANGE)
    for _ in range(60):
        middle = 0.5 * (low + high)
        ratio = cost(10 ** middle) / clean
        if np.isfinite(ratio) and abs(ratio / target - 1) < TOLERANCE:
            break
        if np.isfinite(ratio) and ratio < target:
            low = middle
        else:
            high = middle
    return float(f"{10 ** middle:.3g}")


def measure_sizes(sizes=settings.CH4_INFLATION_SIZES):
    """Cost ratio of the optimal controller per inflation size, at the calibrated intensities.

    Reads ``calibration.json`` and writes ``size_ratios.json`` next to it.
    """
    root = experiment_root(EXPERIMENT)
    levels = json.loads((root / "calibration.json").read_text())["levels"]

    result = {}
    for size in sizes:
        problems = [make_problem(seed, settings.INFLATION_SIZES[size]) for seed in settings.SEEDS]
        clean = mean_cost(problems)

        result[size] = {}
        for channel, entries in levels.items():
            result[size][channel] = {
                level: mean_cost(problems, channel, entry["std"]) / clean
                for level, entry in entries.items() if "target" in entry
            }
            print(f"{size:>8} {channel:>12}: "
                  + " ".join(f"{r:.3f}" for r in result[size][channel].values()), flush=True)

    (root / "size_ratios.json").write_text(json.dumps({
        "inflation": {size: list(settings.INFLATION_SIZES[size]) for size in sizes},
        "ratios": result,
    }, indent=2))


def main():
    base = settings.INFLATION_SIZES["base"]
    problems = [make_problem(seed, base) for seed in settings.SEEDS]

    clean = mean_cost(problems)

    result = {}
    for channel in settings.NOISE_CHANNELS:
        result[channel] = {"none": {"std": 0.0, "ratio": 1.0}}
        for level, target in settings.NOISE_COST_TARGETS.items():
            std = calibrate(
                lambda std: mean_cost(problems, channel, std), target, clean
            )
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

    measure_sizes()


if __name__ == "__main__":
    if sys.argv[1:] == ["sizes"]:
        measure_sizes()
    else:
        main()
