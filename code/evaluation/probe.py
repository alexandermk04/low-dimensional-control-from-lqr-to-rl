"""Per-iteration instrumentation to collect data by both LQR solvers and the RL.

The comparison measures three different timescales:

- **iterations**, which are not comparable across methods but are what a
  solver's own loop counts;
- **environment steps**, every model transition, recorded when
  :meth:`~lqr_base.simple_lqr.LQRBase.apply_control` is applied and counted in
  ``lqr.env_steps``.
- **total time**, total time, excluding the probe's own evaluation work.
"""

import time
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from lqr_base.simple_lqr import LQRDiscrete, true_cost

__all__ = ["IterationRecord", "Probe", "record_iteration", "should_stop"]


@dataclass
class IterationRecord:
    """One sample of a solver's progress.

    Attributes
    ----------
    iteration : int
        Index of the iteration that produced this record.
    env_steps : int
        Plant transitions consumed since :meth:`Probe.start`.
    total_time : float
        Seconds of *solver* time since :meth:`Probe.start`
    cost : float
        Value of the evaluation callback at this iterate
    cost_estimate : float or None
        The solver's internal cost estimate, if tracked.
    rank : int or None
        Rank of the raw gain, if tracked.
    """

    iteration: int
    env_steps: int
    total_time: float
    cost: float
    cost_estimate: float | None = None
    rank: int | None = None


class Probe:
    """Records an :class:`IterationRecord` per solver iteration.

    Parameters
    ----------
    lqr : LQRDiscrete
        The system the solver is running against.
    evaluate : Callable[[np.ndarray], float]
        Maps the current raw gain to the cost to record. Typically
        ``lambda K: expected_true_cost(lqr, K)``.
    every : int, optional
        Record only every ``n``-th iteration. Defaults to ``1``.
    track_rank : bool, optional
        Whether to record ``matrix_rank`` of the raw gain. Defaults to ``True``.

    Attributes
    ----------
    records : list[IterationRecord]
        Everything recorded so far, in order.
    """

    def __init__(
        self,
        lqr: LQRDiscrete,
        evaluate: Callable[[np.ndarray], float],
        every: int = 1,
        track_rank: bool = True,
    ):
        self.lqr = lqr
        self.evaluate = evaluate
        self.every = every
        self.track_rank = track_rank

        self.records: list[IterationRecord] = []

        self._t0: float | None = None
        self._steps_0 = 0
        self._overhead = 0.0

    def start(self):
        """Begin timing. Call immediately before the solver's loop."""
        self._t0 = time.perf_counter()
        self._steps_0 = self.lqr.env_steps
        self._overhead = 0.0
        self.records = []

    def record(
        self, iteration: int, K: np.ndarray, cost_estimate: float | None = None
    ) -> IterationRecord | None:
        """Evaluate the current iterate and append a record.

        Parameters
        ----------
        iteration : int
            Index of the iteration just completed.
        K : np.ndarray
            Current raw gain.
        cost_estimate : float or None, optional
            The solver's own cost estimate for this iterate, if it has one.

        Returns
        -------
        IterationRecord or None
            The record, or ``None`` when this iteration was skipped by
            ``every``.
        """
        if self._t0 is None:
            self.start()

        if iteration % self.every != 0:
            return None

        elapsed = time.perf_counter() - self._t0 - self._overhead

        # Time the evaluation itself and give the seconds back, so that the
        # recorded total time is the solver's and not the instrumentation's.
        eval_start = time.perf_counter()
        cost = float(self.evaluate(K))
        rank = int(np.linalg.matrix_rank(K)) if self.track_rank else None
        self._overhead += time.perf_counter() - eval_start

        record = IterationRecord(
            iteration=iteration,
            env_steps=self.lqr.env_steps - self._steps_0,
            total_time=elapsed,
            cost=cost,
            cost_estimate=None if cost_estimate is None else float(cost_estimate),
            rank=rank,
        )
        self.records.append(record)

        return record

    def last(self) -> IterationRecord | None:
        """Most recent record, or ``None`` if nothing has been recorded yet."""
        return self.records[-1] if self.records else None

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Columns of the learning curve, ready to be stored in an NPZ.

        Returns
        -------
        dict[str, np.ndarray]
            One array per :class:`IterationRecord` field. ``cost_estimate`` and
            ``rank`` become ``NaN`` where they were not recorded.
        """
        if not self.records:
            return {}

        columns = {}
        for field in asdict(self.records[0]):
            values = [getattr(r, field) for r in self.records]
            values = [np.nan if v is None else v for v in values]
            columns[field] = np.asarray(values, dtype=float)

        return columns


def record_iteration(
    settings,
    lqr: LQRDiscrete,
    progress_bar: tqdm,
    iteration: int,
    K: np.ndarray,
    cost_estimate: float | None = None,
    extra: dict | None = None,
    show_rank: bool = False,
) -> IterationRecord | None:
    """Record one solver iteration and refresh its progress bar.

    Parameters
    ----------
    settings : Any
        The solver's settings object. Instrumentation happens only if it
        carries a non-``None`` ``probe`` attribute.
    lqr : LQRDiscrete
        System being solved
    progress_bar : tqdm.tqdm
        Bar to update.
    iteration : int
        Index of the iteration just completed.
    K : np.ndarray
        Current raw gain.
    cost_estimate : float or None, optional
        The solver's own cost estimate, if it has one.
    extra : dict or None, optional
        Additional entries to show on the bar.
    show_rank : bool, optional
        Whether to show the gain's rank 

    Returns
    -------
    IterationRecord or None
        The record, or ``None`` when there is no probe or the iteration was
        skipped.
    """
    probe: Probe | None = getattr(settings, "probe", None)
    postfix: dict = {}
    record = None

    if probe is not None:
        record = probe.record(iteration, K, cost_estimate=cost_estimate)
        if record is not None:
            postfix["True Cost"] = f"{record.cost:.3e}"
            postfix["Env Steps"] = record.env_steps
            if record.rank is not None:
                postfix["Rank"] = record.rank
    else:
        postfix["True Cost"] = f"{true_cost(lqr, K):.3e}"
        if show_rank:
            postfix["Matrix Rank"] = int(np.linalg.matrix_rank(K))

    if extra:
        postfix.update(extra)
    if postfix:
        progress_bar.set_postfix(postfix)

    return record


def should_stop(
    settings, lqr: LQRDiscrete, iteration: int, signal: float, K: np.ndarray
) -> bool:
    """Feed the solver's own cost signal to its stopper, if it has one.

    Parameters
    ----------
    settings : Any
        The solver's settings object; acts only if it carries a ``stopper``.
    lqr : LQRDiscrete
        System being solved, for the step count when there is no probe.
    iteration : int
        Index of the iteration just completed.
    signal : float
        The solver's cost for ``K``: model cost or rollout estimate.
    K : np.ndarray
        The gain ``signal`` belongs to.

    Returns
    -------
    bool
        ``True`` once the solver should stop.
    """
    stopper = getattr(settings, "stopper", None)
    if stopper is None:
        return False

    probe: Probe | None = getattr(settings, "probe", None)
    steps = probe.last().env_steps if probe is not None and probe.last() else lqr.env_steps

    return stopper.update(signal, K, iteration, steps)
