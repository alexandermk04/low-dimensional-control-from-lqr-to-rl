"""Chapter 4: robustness of the controllers trained on the true model.

Run with::

    python -m experiments.ch4_true_model_noise
"""

from experiments.ch4_noise import SIZES, evaluate_grid

EXPERIMENT = "ch4_true_model_noise"

SOURCE = "ch4_true_model"


def main():
    evaluate_grid(SOURCE, EXPERIMENT, SIZES)


if __name__ == "__main__":
    main()
