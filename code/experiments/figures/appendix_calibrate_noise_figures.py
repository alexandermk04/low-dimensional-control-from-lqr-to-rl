"""Table of the calibrated noise intensities.

Reads ``experiment_results/appendix_calibrate_noise/calibration.json``.

Run with::

    python -m experiments.figures.appendix_calibrate_noise_figures
"""

import json

from experiments.appendix_calibrate_noise import EXPERIMENT
from experiments.figures.style import table_dir
from experiments.figures.tables import write_table
from experiments.store import ROOT, MissingExperimentError

# Row label per noise channel
CHANNEL_ROWS = {
    "measurement": r"Measurement $\sigma_{\mathrm{m}}$",
    "control": r"Control $\sigma_{\mathrm{c}}$",
    "parametric": r"Parametric $\sigma_{\mathrm{p}}$",
    "disturbance": r"Disturbance $\sigma_{\mathrm{d}}$",
}


def noise_table(calibration: dict, label: str = "tab:noise_levels") -> str:
    """The standard deviation per channel and level, with the targeted cost ratio."""
    levels = calibration["levels"]
    names = list(next(iter(levels.values())))

    ratios = [
        entry["ratio"] / entry["target"] - 1
        for channel in levels.values() for entry in channel.values() if "target" in entry
    ]
    worst = max(abs(r) for r in ratios)

    targets = [
        levels[next(iter(levels))][name].get("target", 1) for name in names
    ]

    lines = [
        r"\begin{tabular}{@{}l" + "r" * len(names) + "@{}}",
        r"\toprule",
        " & ".join(["Channel"] + [rf"\emph{{{name}}}" for name in names]) + r" \\",
        r"\midrule",
        " & ".join([r"Cost ratio $\rho_{\ell}$"] + [f"${t:g}$" for t in targets]) + r" \\",
        r"\midrule",
    ]
    for channel, entries in levels.items():
        lines.append(" & ".join(
            [CHANNEL_ROWS.get(channel, channel)]
            + [f"${entries[name]['std']:.3g}$" for name in names]
        ) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    caption = (
        r"Standard deviation per noise channel and intensity, calibrated to the "
        r"cost ratio $\rho_{\ell}$ of \eqref{eq:noise-calibration}. Every value "
        rf"reaches its ratio within ${100 * worst:.1f}\,\%$."
    )

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        *lines,
        r"\end{table}",
    ])


def main():
    path = ROOT / EXPERIMENT / "calibration.json"
    if not path.exists():
        raise MissingExperimentError(
            f"No calibration at {path}. Run `python -m experiments.{EXPERIMENT}` first."
        )

    write_table(noise_table(json.loads(path.read_text())), "noise_levels", EXPERIMENT)
    print("wrote tables  to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
