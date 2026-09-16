"""Matplotlib configured to render through LaTeX.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt  # noqa: E402

import settings  # noqa: E402

__all__ = [
    "TEXT_WIDTH",
    "PGF_PREAMBLE_GUARD",
    "figure_size",
    "COLORS",
    "color_for",
    "new_figure",
    "save",
    "figure_dir",
    "table_dir",
]

# Copied from thesis.tex, in inches.
TEXT_WIDTH = 5.8

BASE_FONT_SIZE = 12

# Colour-blind safe & greyscale valid colors
COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#000000",  # black for baseline
)

# The optimum should read the same in every figure it appears in.
BASELINE_COLOR = "#000000"

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.family": "serif",
    "font.size": BASE_FONT_SIZE - 2,
    "axes.labelsize": BASE_FONT_SIZE - 2,
    "axes.titlesize": BASE_FONT_SIZE - 1,
    "legend.fontsize": BASE_FONT_SIZE - 4,
    "xtick.labelsize": BASE_FONT_SIZE - 4,
    "ytick.labelsize": BASE_FONT_SIZE - 4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
})


def figure_size(width: float = 1.0, aspect: float = 0.62) -> tuple[float, float]:
    """A figure size in inches, as a fraction of the text width.

    Parameters
    ----------
    width : float, optional
        Fraction of ``\\textwidth``. ``1.0`` is full width, ``0.48`` fits two
        side by side.
    aspect : float, optional
        Height over width. Defaults to roughly the golden ratio.

    Returns
    -------
    tuple[float, float]
        ``(width, height)`` in inches.
    """
    inches = TEXT_WIDTH * width
    return (inches, inches * aspect)


def color_for(index: int, name=None) -> str:
    """Colour of the ``index``-th group, with the baseline always black.

    Parameters
    ----------
    index : int
        Position of the group.
    name : Any, optional
        The group's key. Only used to recognise the baseline.
    """
    if name is not None and str(name).startswith("baseline"):
        return BASELINE_COLOR
    return COLORS[index % len(COLORS)]


def new_figure(width: float = 1.0, aspect: float = 0.62, **kwargs):
    """``plt.subplots`` at a scaled size."""
    return plt.subplots(figsize=figure_size(width, aspect), **kwargs)


PGF_PREAMBLE_GUARD = (
    "%% Written by experiments/figures/style.py so this file can be \\input\n"
    "%% into any document that loads pgf.\n"
    "\\providecommand{\\mathdefault}[1]{#1}%\n"
)


def figure_dir(experiment: str) -> Path:
    """Where ``.pgf`` files go, created if missing."""
    path = Path(settings.GENERATED_ROOT) / experiment / settings.FIGURE_OUTPUT
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_dir(experiment: str) -> Path:
    """Where ``.tex`` table fragments go, created if missing."""
    path = Path(settings.GENERATED_ROOT) / experiment / settings.TABLE_OUTPUT
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(figure, name: str, experiment: str, also_pdf: bool = True) -> Path:
    """Write a figure as ``.pgf``, and optionally a ``.pdf`` for previews.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
    name : str
        File stem. 
    experiment : str
        Name for the output directory.
    also_pdf : bool, optional
        Whether to create the PDF version.

    Returns
    -------
    Path
        The ``.pgf`` file.
    """
    directory = figure_dir(experiment)
    path = directory / f"{name}.pgf"

    figure.savefig(path, bbox_inches="tight")
    path.write_text(PGF_PREAMBLE_GUARD + path.read_text())

    if also_pdf:
        figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")

    plt.close(figure)

    return path
