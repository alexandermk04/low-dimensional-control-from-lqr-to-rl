"""Summaries of a learning curve.
"""

import numpy as np

from evaluation.probe import IterationRecord

__all__ = [
    "suboptimality",
    "time_to_threshold",
    "converged",
    "policy_complexity_linear",
    "policy_complexity_mlp",
]


def suboptimality(cost: float, j_star: float) -> float:
    """Relative excess over the optimum, ``(J - J*) / J*``.

    Parameters
    ----------
    cost : float
        Cost achieved.
    j_star : float
        Optimal cost.

    Returns
    -------
    float
        The relative gap; ``np.inf`` if ``cost`` is infinite.
    """
    if not np.isfinite(cost):
        return np.inf

    return float((cost - j_star) / j_star)


def time_to_threshold(
    records: list[IterationRecord],
    j_star: float,
    tol: float = 0.05,
    require_sustained: bool = True,
) -> dict[str, float | None]:
    """When a run first got and potentially stayed within ``tol`` of the optimum.

    Parameters
    ----------
    records : list[IterationRecord]
        The learning curve, in order.
    j_star : float
        Optimal cost, the reference the threshold is relative to.
    tol : float, optional
        Relative tolerance. ``0.05`` means "within 5 % of optimal".
    require_sustained : bool, optional
        If ``True`` (default), the crossing only counts when every later record
        also stays within the threshold.

    Returns
    -------
    dict[str, float | None]
        ``iterations``, ``env_steps`` and ``total_time`` at the crossing, and
        the ``cost`` there. Every value is ``None`` if the run never
        converged.
    """
    threshold = j_star * (1.0 + tol)

    crossing = None
    for index, record in enumerate(records):
        if not (np.isfinite(record.cost) and record.cost <= threshold):
            continue
        if require_sustained and not all(
            np.isfinite(later.cost) and later.cost <= threshold
            for later in records[index:]
        ):
            continue
        crossing = record
        break

    if crossing is None:
        return {"iterations": None, "env_steps": None, "total_time": None, "cost": None}

    return {
        "iterations": float(crossing.iteration),
        "env_steps": float(crossing.env_steps),
        "total_time": float(crossing.total_time),
        "cost": float(crossing.cost),
    }


def converged(
    records: list[IterationRecord], tol_rel: float = 1e-3, window: int = 10
) -> bool:
    """Whether the cost has converged by the end of the run.

    Parameters
    ----------
    records : list[IterationRecord]
        The learning curve.
    tol_rel : float, optional
        Maximum relative spread across the window. Defaults to ``1e-3``.
    window : int, optional
        Number of trailing records to look at. Defaults to ``10``.

    Returns
    -------
    bool
        ``True`` when the last ``window`` costs are finite and vary by less
        than ``tol_rel`` relative to their mean.
    """
    if len(records) < window:
        return False

    tail = np.array([r.cost for r in records[-window:]], dtype=float)

    if not np.all(np.isfinite(tail)) or np.mean(tail) == 0:
        return False

    return bool((tail.max() - tail.min()) / abs(np.mean(tail)) < tol_rel)


def policy_complexity_linear(K: np.ndarray, tol: float | None = None) -> dict[str, int]:
    """Free-parameter count of a linear policy, given its rank.

    Parameters
    ----------
    K : np.ndarray
        Raw gain in the controller's space, of shape ``(q, p)``.
    tol : float or None, optional
        Singular-value tolerance forwarded to :func:`numpy.linalg.matrix_rank`.

    Returns
    -------
    dict[str, int]
        ``rank``, ``n_params`` (the cheaper of the factored and the dense
        count) and ``n_params_dense`` (``q * p``, what an unstructured policy
        costs).
    """
    rank = int(np.linalg.matrix_rank(K) if tol is None else np.linalg.matrix_rank(K, tol=tol))
    q, p = K.shape

    return {
        "rank": rank,
        "n_params": min(rank * (q + p), q * p),
        "n_params_dense": q * p,
    }
