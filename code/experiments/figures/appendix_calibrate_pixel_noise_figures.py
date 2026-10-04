"""Table of the calibrated noise intensities of the camera environment.

Run with::

    python -m experiments.figures.appendix_calibrate_pixel_noise_figures
"""

import json

import settings
from experiments.appendix_calibrate_pixel_noise import EXPERIMENT
from experiments.figures.appendix_calibrate_noise_figures import noise_table
from experiments.figures.style import table_dir
from experiments.figures.tables import write_table
from experiments.store import ROOT, MissingExperimentError


def main():
    levels = {}
    for channel in settings.NOISE_CHANNELS:
        path = ROOT / EXPERIMENT / f"calibration_{channel}.json"
        if not path.exists():
            raise MissingExperimentError(
                f"No calibration at {path}. Run `python -m experiments.{EXPERIMENT}` first."
            )
        levels.update(json.loads(path.read_text())["levels"])

    write_table(
        noise_table({"levels": levels}, label="tab:pixel-noise-levels"),
        "pixel_noise_levels", EXPERIMENT,
    )
    print("wrote tables  to", table_dir(EXPERIMENT))

    print("\nCH6_NOISE_STD = {")
    for channel, entries in levels.items():
        print(f'    "{channel}": {{')
        for level, entry in entries.items():
            print(f'        "{level}": {entry["std"]},')
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
