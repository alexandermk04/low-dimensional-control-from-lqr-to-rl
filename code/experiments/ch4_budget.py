"""Data sample budgets for the solvers.

Shared module, to use the same ones throughout.

Is determined by experiments which will be shown in the appendix, to simplify
the logic of experiments.
"""

import settings
from experiments.settings_guard import require_per_size

__all__ = ["SAMPLE_SET_BY", "IDENTIFICATION_SET_BY", "budget", "condition"]

SAMPLE_SET_BY = "python -m experiments.appendix_sample_size"

IDENTIFICATION_SET_BY = "python -m experiments.appendix_identification_length"


def budget(solver: str, size: str) -> tuple[int | None, int | None]:
    """The training budget one solver is given at one inflation size.

    Parameters
    ----------
    solver : str
        Key into :data:`experiments.registry.SOLVER_REGISTRY`.
    size : str
        Name of an entry of ``settings.INFLATION_SIZES``.

    Returns
    -------
    tuple[int or None, int or None]
        ``(rollout_length, n_trajectories)``. The second is ``None`` for an
        identification solver, both are ``None`` for a solver working without data.
    """
    if solver in settings.IDENTIFICATION_SOLVERS:
        chosen = require_per_size(
            settings.CHOSEN_IDENTIFICATION,
            "CHOSEN_IDENTIFICATION",
            IDENTIFICATION_SET_BY,
            sizes=(size,),
        )
        trajectory_length, _ = chosen[size]
        return trajectory_length, None

    if solver in settings.ROLLOUT_SOLVERS:
        chosen = require_per_size(
            settings.CHOSEN_SAMPLE_CELL, "CHOSEN_SAMPLE_CELL", SAMPLE_SET_BY,
            sizes=(size,),
        )
        return chosen[size]

    return None, None


def condition(solver: str, size: str) -> dict:
    """Add the chosen Hankel horizon for identification solvers to the condition dict.

    Parameters
    ----------
    solver : str
        Key into :data:`experiments.registry.SOLVER_REGISTRY`.
    size : str
        Name of an entry of ``settings.INFLATION_SIZES``.

    Returns
    -------
    dict
        Passed to :attr:`experiments.problem.RunSpec.condition`.
    """
    if solver in settings.IDENTIFICATION_SOLVERS:
        require_per_size(
            settings.CHOSEN_IDENTIFICATION,
            "CHOSEN_IDENTIFICATION",
            IDENTIFICATION_SET_BY,
            sizes=(size,),
        )
        _, horizon = settings.CHOSEN_IDENTIFICATION[size]
        return {"size": size, "hankel": horizon}

    return {"size": size}
