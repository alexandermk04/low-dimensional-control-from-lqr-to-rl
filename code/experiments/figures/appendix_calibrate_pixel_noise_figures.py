"""Table of the noise intensities of the camera environment, and a view of the measurement noise.

Run with::

    python -m experiments.figures.appendix_calibrate_pixel_noise_figures
"""

import json

import numpy as np

import settings
from experiments.appendix_calibrate_pixel_noise import CALIBRATED_CHANNELS, EXPERIMENT
from experiments.figures.appendix_calibrate_noise_figures import noise_table
from experiments.figures.style import figure_dir, new_figure, save, table_dir
from experiments.figures.tables import write_table
from experiments.store import ROOT, MissingExperimentError

# Shown next to the clean frames
PREVIEW_LEVELS = ("standard", "divergent")


def measurement_preview():
    """The two stacked frames without measurement noise and at :data:`PREVIEW_LEVELS`."""
    from experiments.noise_builder import build_noise
    from experiments.registry import _pixel_setup
    from lqr_base.thruster_ring import ThrusterOraclePolicy
    from reinforcement_learning.pixel_env import build_pixel_env
    from seeding import sub_seed

    K, _, horizon, _ = _pixel_setup(None)
    levels = ("none",) + PREVIEW_LEVELS

    figure, axes = new_figure(aspect=0.22, ncols=len(levels))
    for ax, level in zip(axes, levels):
        env = build_pixel_env(sub_seed(0, "pixel-preview"), horizon=horizon)
        std = settings.CH6_NOISE_STD["measurement"][level]
        env.lqr.setup_noise(build_noise(("measurement",), level, {"measurement": std}))

        oracle = ThrusterOraclePolicy(K, env.c)
        observation, info = env.reset()
        for _ in range(3):
            observation, _, _, _, info = env.step(oracle(info["state"]))

        separator = np.full((observation.shape[1], 2), 128, dtype=np.uint8)
        ax.imshow(np.hstack([observation[0], separator, observation[1]]), cmap="gray",
                  vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_title(r"\emph{" + level + "}" if level != "none" else "without noise")
        ax.set_xticks([])
        ax.set_yticks([])

    save(figure, "pixel_measurement_noise", EXPERIMENT)


def main():
    levels = {}
    for channel in CALIBRATED_CHANNELS:
        path = ROOT / EXPERIMENT / f"calibration_{channel}.json"
        if not path.exists():
            raise MissingExperimentError(
                f"No calibration at {path}. Run `python -m experiments.{EXPERIMENT}` first."
            )
        levels.update(json.loads(path.read_text())["levels"])

    levels["measurement"] = {
        level: {"std": std} for level, std in settings.CH6_NOISE_STD["measurement"].items()
    }

    write_table(
        noise_table(
            {"levels": levels}, label="tab:pixel-noise-levels",
            note=(
                r"Measurement noise is set by hand and not calibrated."
            ),
        ),
        "pixel_noise_levels", EXPERIMENT,
    )
    print("wrote tables  to", table_dir(EXPERIMENT))

    measurement_preview()
    print("wrote figures to", figure_dir(EXPERIMENT))

    print("\nCH6_NOISE_STD = {")
    for channel in CALIBRATED_CHANNELS:
        print(f'    "{channel}": {{')
        for level, entry in levels[channel].items():
            print(f'        "{level}": {entry["std"]},')
        print("    },")


if __name__ == "__main__":
    main()
