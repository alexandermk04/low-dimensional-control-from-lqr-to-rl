"""Summaries of a learning curve.
"""

import numpy as np

from evaluation.probe import IterationRecord

__all__ = [
    "suboptimality",
    "time_to_threshold",
    "converged",
    "Plateau",
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


class Plateau:
    """Stopping rule and best-iterate selection.

    Parameters
    ----------
    rel_tol : float
        Minimum relative improvement over ``patience`` updates that counts.
    patience : int
        Stop once the smoothed signal is not ``rel_tol`` lower than it was
        ``patience`` updates ago.
    window : int, optional
        Moving-mean window the signal is smoothed over. Defaults to ``1``.

    Attributes
    ----------
    best : float
        Best smoothed signal so far.
    best_payload : Any
        What was passed with the best signal, e.g. the gain or policy weights.
    best_iteration, best_steps : int or None
        Iteration and environment steps at the best signal.
    converged : bool
        Whether the stopping rule was applied.
    """

    def __init__(self, rel_tol: float, patience: int, window: int = 1):
        self.rel_tol = rel_tol
        self.patience = patience
        self.window = window

        self._history: list[float] = []
        self._smoothed: list[float] = []
        self.best = np.inf
        self.best_payload = None
        self.best_iteration: int | None = None
        self.best_steps: int | None = None
        self.converged = False

    def update(self, signal: float, payload, iteration: int, steps: int) -> bool:
        """Add one signal value; return ``True`` once the solver should stop."""
        self._history.append(float(signal))
        smoothed = float(np.mean(self._history[-self.window:]))

        # The trend only counts from the last unstable iterate on
        if not np.isfinite(smoothed):
            self._smoothed = []
        self._smoothed.append(smoothed)

        if self.best_payload is None or smoothed < self.best:
            self.best = smoothed
            self.best_payload = payload
            self.best_iteration = int(iteration)
            self.best_steps = int(steps)

        if len(self._smoothed) <= self.patience:
            return False

        before = self._smoothed[-1 - self.patience]
        improved = not np.isfinite(before) or smoothed < before - self.rel_tol * abs(before)

        self.converged = not improved
        return self.converged

    def extra(self) -> dict:
        """Summary entries describing where the run stopped and what it kept."""
        return {
            "converged": self.converged,
            "best_iteration": self.best_iteration,
            "best_env_steps": self.best_steps,
        }


def policy_complexity_linear(K: np.ndarray, tol: float | None = None) -> dict[str, int]:
    """Storage cost of a linear policy, given its rank.

    Parameters
    ----------
    K : np.ndarray
        Raw gain in the controller's space, of shape ``(q, p)``.
    tol : float or None, optional
        Singular-value tolerance forwarded to :func:`numpy.linalg.matrix_rank`.

    Returns
    -------
    dict[str, int]
        ``rank``, ``n_params`` (equals ``min(q*p, r*(q + p + 1))``) 
        and ``n_params_dense`` (``q * p``).
    """
    rank = int(np.linalg.matrix_rank(K) if tol is None else np.linalg.matrix_rank(K, tol=tol))
    q, p = K.shape

    return {
        "rank": rank,
        # Storage space of SVD factors: U (q*r), the diagonal of Sigma (r) and
        # V^T (r*p), so r*(q + p + 1)
        "n_params": min(rank * (q + p + 1), q * p),
        "n_params_dense": q * p,
    }

def policy_complexity_mlp(model, frozen_extractor: bool = False,
                          n_frozen_extra: int = 0) -> dict[str, int]:
    """Parameter count of a Stable-Baselines3 actor, its feature extractor included.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        A trained model.
    frozen_extractor : bool, optional
        Whether the extractor is frozen post training.
    n_frozen_extra : int, optional
        Fixed parameters outside the model.

    Returns
    -------
    dict[str, int]
        ``rank`` (always ``None``-like ``-1``), ``n_params`` and ``n_params_dense``, both the actor's parameter
        count, and ``n_params_frozen``.

    Raises
    ------
    AttributeError
        If no actor-like submodule can be found on the policy.
    """
    policy = model.policy

    if not hasattr(policy, "action_net"):               # PPO
        raise AttributeError(
            f"Cannot locate an actor on {type(policy).__name__} to count parameters."
        )

    def count(*modules) -> int:
        return sum(
            int(parameter.numel())
            for module in modules
            for parameter in module.parameters()
        )

    extractor = count(policy.pi_features_extractor)
    frozen = (extractor if frozen_extractor else 0) + int(n_frozen_extra)
    total = extractor + count(policy.mlp_extractor.policy_net, policy.action_net) + int(n_frozen_extra)

    return {"rank": -1, "n_params": total, "n_params_dense": total, "n_params_frozen": frozen}
