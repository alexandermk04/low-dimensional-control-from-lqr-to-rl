"""Loading a trained controller from a stored run.
"""

from pathlib import Path

from evaluation.metrics import LinearPolicy, Policy
from evaluation.records import RunRecord, record_dir
from experiments.store import ROOT

__all__ = ["MissingPolicyError", "load_policy"]


class MissingPolicyError(RuntimeError):
    """A stored run carries nothing that can be turned back into a policy."""


def load_policy(record: RunRecord, root: Path | str = ROOT) -> Policy:
    """Rebuild the controller a stored run produced.

    Parameters
    ----------
    record : RunRecord
        As loaded from disk, with ``arrays`` populated.
    root : Path or str, optional
        Results root the record was written under.

    Returns
    -------
    Policy
        Callable mapping an observation to a control, ready for
        :func:`~experiments.evaluate.evaluate_policy`.

    Raises
    ------
    MissingPolicyError
        If the run stores neither a gain nor a policy.
    """
    spec = record.policy or {}
    kind = spec.get("kind")

    if kind is None:
        raise MissingPolicyError(
                        f"{record.run_id} contains no policy kind. "
                        f"Please rerun the experiments."
                    )

    if kind == "linear":
        gain = record.arrays.get("K_final")
        if gain is None or not gain.size:
            raise MissingPolicyError(
                f"{record.run_id} contains no K_final array."
            )
        return LinearPolicy(gain)

    if kind == "sb3":
        from reinforcement_learning.solvers import ALGORITHMS, SB3Policy

        algo = spec.get("algo")
        if algo not in ALGORITHMS:
            raise MissingPolicyError(
                f"{record.run_id} names algorithm {algo!r}, which is not one "
                f"of {sorted(ALGORITHMS)}."
            )

        path = record_dir(record, root) / spec.get("file", "policy.zip")
        if not path.exists():
            raise MissingPolicyError(
                f"{record.run_id} names a policy file at {path} that is not "
                f"there."
            )

        model = ALGORITHMS[algo].load(path, device="cpu")

        return SB3Policy(model, deterministic=spec.get("deterministic", True))

    raise MissingPolicyError(
        f"{record.run_id} stores no reloadable policy "
        f"(solver {record.solver!r}, descriptor {spec!r})."
    )
