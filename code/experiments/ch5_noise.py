"""Chapter 5: robustness in the inflated dimensions.

Run with::

    python -m experiments.ch5_noise
"""

import settings
from experiments.ch4_noise import evaluate_grid
from experiments.settings_guard import require_choice

EXPERIMENT = "ch5_noise"

SOURCE = "ch5_solver_comparison"

# The chosen inflation size for comparison
SIZES = (settings.CH5_REFERENCE_INFLATION,)


def main():
    require_choice(
        settings.CH5_REFERENCE_INFLATION, "CH5_REFERENCE_INFLATION",
        settings.INFLATION_SIZES, "pick one of settings.CH5_INFLATION_SIZES",
    )

    evaluate_grid(SOURCE, EXPERIMENT, SIZES)


if __name__ == "__main__":
    main()
