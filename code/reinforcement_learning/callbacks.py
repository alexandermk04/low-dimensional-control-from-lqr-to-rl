"""Instrumentation for an RL run, emitting the same records as an LQR solver.

A :class:`~evaluation.probe.Probe` samples a solver once per iteration; this
callback samples an RL algorithm once per ``eval_freq`` environment steps.
"""

import copy
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
    env_steps_offset : int, optional
        Added to every recorded ``env_steps``.
    env_steps_scale : int, optional
        Multiplies the algorithm's step count before the offset is added.
        Used for agents acting slower than the environment. Defaults to ``1``.
    stopper : evaluation.analysis.Plateau or None, optional
        Fed at every sample with the mean cost of the training episodes
        finished since the previous one.

    Attributes
    ----------
    records : list[IterationRecord]
        The learning curve, in order.
    """

    def __init__(self, evaluate, eval_freq: int, verbose: int = 0,
                 env_steps_offset: int = 0, env_steps_scale: int = 1,
                 stopper=None):
        super().__init__(verbose)

        if env_steps_scale < 1:
            raise ValueError(f"env_steps_scale must be positive, got {env_steps_scale}.")

        self.evaluate = evaluate
        self.eval_freq = eval_freq
        self.env_steps_offset = int(env_steps_offset)
        self.env_steps_scale = int(env_steps_scale)
        self.stopper = stopper

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
        self._returns: list[float] = []
        self._stop = False
        self._sample()

    def _sample(self) -> IterationRecord:
        """Evaluate the current policy and append a record."""
        elapsed = time.perf_counter() - self._t0 - self._overhead

        start = time.perf_counter()
        cost, rank = self.evaluate(self.model)
        self._overhead += time.perf_counter() - start

        record = IterationRecord(
            iteration=len(self.records),
            env_steps=int(self.num_timesteps) * self.env_steps_scale + self.env_steps_offset,
            total_time=elapsed,
            cost=float(cost),
            cost_estimate=None,
            rank=None if rank is None else int(rank),
        )
        self.records.append(record)
        self._next_eval = self.num_timesteps + self.eval_freq

        if self.stopper is not None and self._returns:
            signal = -float(np.mean(self._returns))
            self._stop = self.stopper.update(
                signal, copy.deepcopy(self.model.policy.state_dict()),
                record.iteration, record.env_steps,
            )
        self._returns = []

        if self.verbose:
            print(f"[probe] steps={record.env_steps:>9d}  "
                  f"J={record.cost:.4e}  t={record.total_time:6.1f}s")

        return record

    def _on_step(self) -> bool:
        """Sample whenever the step budget passes the next evaluation point."""
        for info in self.locals.get("infos", ()):
            if "episode" in info:
                self._returns.append(float(info["episode"]["r"]))

        if self.num_timesteps >= self._next_eval:
            self._sample()
        return not self._stop

    def restore_best(self) -> None:
        """Load the best policy weights the stopper kept into the model."""
        if self.stopper is not None and self.stopper.best_payload is not None:
            self.model.policy.load_state_dict(self.stopper.best_payload)

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
