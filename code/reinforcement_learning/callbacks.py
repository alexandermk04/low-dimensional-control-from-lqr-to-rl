"""Instrumentation for an RL run, emitting the same records as an LQR solver.

A :class:`~evaluation.probe.Probe` samples a solver once per iteration; this
callback samples an RL algorithm once per ``eval_freq`` environment steps.
"""

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.probe import IterationRecord

__all__ = ["ProbeCallback"]


class ProbeCallback(BaseCallback):
    """Record cost against sample budget and total time during training.

    Parameters
    ----------
    evaluate : Callable[[stable_baselines3.common.base_class.BaseAlgorithm], tuple[float, int | None]]
        Given the model, returns ``(cost, rank)``. ``rank`` may be ``None`` for
        a policy that has no gain matrix.
    eval_freq : int
        Evaluate every this many environment steps.
    verbose : int, optional
        Passed to :class:`~stable_baselines3.common.callbacks.BaseCallback`.

    Attributes
    ----------
    records : list[IterationRecord]
        The learning curve, in order.
    """

    def __init__(self, evaluate, eval_freq: int, verbose: int = 0):
        super().__init__(verbose)

        self.evaluate = evaluate
        self.eval_freq = eval_freq

        self.records: list[IterationRecord] = []

        self._t0: float | None = None
        self._overhead = 0.0
        self._next_eval = 0

    def _on_training_start(self) -> None:
        """Start the clock and take the first sample, at zero steps."""
        self._t0 = time.perf_counter()
        self._overhead = 0.0
        self.records = []
        self._next_eval = 0
        self._sample()

    def _sample(self) -> IterationRecord:
        """Evaluate the current policy and append a record."""
        elapsed = time.perf_counter() - self._t0 - self._overhead

        start = time.perf_counter()
        cost, rank = self.evaluate(self.model)
        self._overhead += time.perf_counter() - start

        record = IterationRecord(
            iteration=len(self.records),
            env_steps=int(self.num_timesteps),
            total_time=elapsed,
            cost=float(cost),
            cost_estimate=None,
            rank=None if rank is None else int(rank),
        )
        self.records.append(record)
        self._next_eval = self.num_timesteps + self.eval_freq

        if self.verbose:
            print(f"[probe] steps={record.env_steps:>9d}  "
                  f"J={record.cost:.4e}  t={record.total_time:6.1f}s")

        return record

    def _on_step(self) -> bool:
        """Sample whenever the step budget passes the next evaluation point."""
        if self.num_timesteps >= self._next_eval:
            self._sample()
        return True

    def _on_training_end(self) -> None:
        """Take a final sample."""
        if not self.records or self.records[-1].env_steps < self.num_timesteps:
            self._sample()

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Learning-curve columns, matching :meth:`evaluation.probe.Probe.to_arrays`."""
        from dataclasses import asdict

        if not self.records:
            return {}

        columns = {}
        for name in asdict(self.records[0]):
            values = [getattr(r, name) for r in self.records]
            columns[name] = np.asarray(
                [np.nan if v is None else v for v in values], dtype=float
            )
        return columns
