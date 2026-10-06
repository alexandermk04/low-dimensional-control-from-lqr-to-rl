"""Subspace identification of an unknown plant, for the model-based solvers.

The model-based solvers in :mod:`lqr_solvers.mb_policy_gradient` and
:mod:`lqr_solvers.structured_policy_iteration` read the model from the plant.
This module identifies ``A``, ``B`` and the observation map ``C`` from *data*
instead, so the same solvers can run on a surrogate.

References
----------
.. [1] P. Van Overschee, B. De Moor, "N4SID: Subspace algorithms for the
       identification of combined deterministic-stochastic systems",
       Automatica 30(1), 1994.
       Called through [2]_.
.. [2] G. Armenise et al., "SIPPY: Systems Identification Package for Python".
       https://github.com/CPCLAB-UNIPI/SIPPY
"""

from dataclasses import dataclass

import numpy as np

from basetypes import StateSpaceModel
from lqr_base.simple_lqr import LQRBase, LQRDiscrete, LQRDiscreteSettings

__all__ = [
    "IdentificationError",
    "IdentificationSettings",
    "IdentifiedModel",
    "effective_max_order",
    "collect_identification_data",
    "identify",
    "to_surrogate",
]


class IdentificationError(RuntimeError):
    """Identification did not produce a usable model.
    """


@dataclass
class IdentificationSettings:
    """How to excite the plant and how to read the result.

    Attributes
    ----------
    trajectory_length : int
        ``T``, the number of steps of the single identification trajectory.
    excitation : float
        Standard deviation of the exploration input added on top of ``-K_0 y``.
    seed_seq : np.random.SeedSequence
        Seed Sequence for the exploration input.
    max_order : int
        Largest state dimension the order search may return. Must be an
        ``int`` as SIPPY rejects numpy ints.
    horizon : int
        Past and future horizon of the block Hankel matrices, SIPPY's ``SS_f``
        and ``SS_p``.
    order_threshold : float
        Relative singular-value cutoff for the order search.
    fixed_order : int or None, optional
        Skip the order search and identify at this order.
    """

    trajectory_length: int
    excitation: float
    seed_seq: np.random.SeedSequence
    max_order: int = 10
    horizon: int = 10
    order_threshold: float = 1e-3
    fixed_order: int | None = None


@dataclass
class IdentifiedModel:
    """A plant reconstructed from data, in its own coordinates.

    The identified system is ``z' = A z + B v`` with observation ``y = C z``,
    where ``z`` is the identified state and ``v`` the full controller output.

    Attributes
    ----------
    A : np.ndarray
        Identified state matrix of shape ``(n_hat, n_hat)``
    B : np.ndarray
        Identified input matrix of shape ``(n_hat, q)``
    decode : np.ndarray
        ``(n_hat, p)``, mapping an observation to the identified state. Equal
        to ``C'``, as ``C`` has orthonormal columns.
    n_hat : int
        Estimated state dimension.
    effective_max_order : int
        The order bound that was enforced, see :func:`effective_max_order`.
    dt : float
        Sampling interval the model was identified at.
    """

    A: np.ndarray
    B: np.ndarray
    decode: np.ndarray
    n_hat: int
    effective_max_order: int
    dt: float


def collect_identification_data(
    plant: LQRBase,
    K_0: np.ndarray,
    settings: IdentificationSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive the plant for one excited trajectory and record what came back.

    Parameters
    ----------
    plant : LQRBase
        System to identify. Only :meth:`~lqr_base.simple_lqr.LQRBase.observe`
        and :meth:`~lqr_base.simple_lqr.LQRBase.apply_control` are used.
    K_0 : np.ndarray
        Stabilizing starting gain in the controller's space, of shape ``(q, p)``.
    settings : IdentificationSettings
        Settings for this run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Y, V)`` with ``Y`` of shape ``(p, T)`` and ``V`` of shape
        ``(q, T)``.
    """
    rng = np.random.default_rng(settings.seed_seq)

    plant.reset()

    observations = []
    actions = []

    for _ in range(settings.trajectory_length):
        y = plant.observe()
        v = -K_0 @ y + settings.excitation * rng.standard_normal(plant.action_dim)

        step = plant.apply_control(v)

        observations.append(step.y_t)
        actions.append(step.v_t)

    return np.asarray(observations).T, np.asarray(actions).T


def _run_n4sid(Y: np.ndarray, V: np.ndarray, dt: float, settings: IdentificationSettings):
    """Hand the trajectory to SIPPY and return its raw realization.
    """
    from sippy_unipi import system_identification

    if settings.fixed_order is not None:
        order_kwargs = {"SS_fixed_order": int(settings.fixed_order)}
    else:
        order_kwargs = {
            "SS_max_order": int(settings.max_order),
            "SS_threshold": float(settings.order_threshold),
        }

    try:
        return system_identification(
            Y, V, "N4SID",
            tsample=dt,
            SS_f=int(settings.horizon),
            SS_p=int(settings.horizon),
            **order_kwargs,
        )
    except np.linalg.LinAlgError as failure:
        raise IdentificationError(
            f"The subspace decomposition did not converge on a "
            f"{Y.shape[0]}-dimensional observation with horizon "
            f"{settings.horizon}: {failure}. Lower "
            "IdentificationSettings.horizon, or lengthen the trajectory."
        ) from failure


def _normalize_realization(A, B, C):
    """Rotate the realization so ``C`` has orthonormal columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(A, B, C)`` in the normalized basis.
    """
    Q_c, R_c = np.linalg.qr(C)

    signs = np.sign(np.diag(R_c))
    signs[signs == 0] = 1.0
    Q_c = Q_c * signs
    R_c = R_c * signs[:, None]

    return R_c @ A @ np.linalg.inv(R_c), R_c @ B, Q_c


def effective_max_order(settings: IdentificationSettings) -> int:
    """The order bound actually in force, might have been lowered due to the Hankel horizon.
    """
    if settings.fixed_order is not None:
        return int(settings.fixed_order)

    return min(int(settings.max_order), int(settings.horizon))


def identify(
    plant: LQRBase,
    K_0: np.ndarray,
    settings: IdentificationSettings,
) -> IdentifiedModel:
    """Identify a plant from one excited trajectory.

    Parameters
    ----------
    plant : LQRBase
        System to identify.
    K_0 : np.ndarray
        Stabilizing starting gain in the controller's space.
    settings : IdentificationSettings
        Trajectory length, excitation size and the order search.

    Returns
    -------
    IdentifiedModel

    Raises
    ------
    IdentificationError
        If the order search saturated at ``max_order``, meaning the
        trajectory was too short to determine the model.
    """
    Y, V = collect_identification_data(plant, K_0, settings)

    identified = _run_n4sid(Y, V, plant.dt, settings)

    A, B, C = _normalize_realization(identified.A, identified.B, identified.C)
    n_hat = A.shape[0]

    bound = effective_max_order(settings)

    if settings.fixed_order is None and n_hat >= bound:
        asked = int(settings.max_order)
        clamped = "" if bound == asked else (
            f" (set max_order to {asked}, clamped to the Hankel horizon "
            f"{settings.horizon})"
        )
        raise IdentificationError(
            f"Order search saturated at {bound}{clamped} from "
            f"T={settings.trajectory_length} samples of a "
            f"{plant.observation_dim}-dimensional observation. The trajectory "
            "is too short to determine the model; raise "
            "IdentificationSettings.trajectory_length."
        )

    # C is orthonormal after normalization, so its transpose is its
    # pseudo-inverse
    return IdentifiedModel(
        A=A,
        B=B,
        decode=C.T,
        n_hat=n_hat,
        effective_max_order=bound,
        dt=plant.dt,
    )


class _IdentifiedLQR(LQRDiscrete):
    """An identified model in the controller's space.

    Ensures identified matrices are not discretized again.
    """

    def __init__(self, settings: LQRDiscreteSettings):
        LQRBase.__init__(self, settings)
        self.dt = settings.dt


def to_surrogate(
    model: IdentifiedModel,
    Q: np.ndarray,
    R: np.ndarray,
    seed_seq: np.random.SeedSequence,
) -> LQRDiscrete:
    """Wrap an identified model as a plant.

    Parameters
    ----------
    model : IdentifiedModel
        The identified system.
    Q : np.ndarray
        The known state cost on the true state, a multiple of the identity,
        as a diagonal ``(n,)``.
    R : np.ndarray
        The known cost on the controller output, ``(q,)`` or ``(q, q)``.
    seed_seq : np.random.SeedSequence
        Seeds the surrogate's initial state.

    Returns
    -------
    LQRDiscrete
        A surrogate on ``(A, B)`` with ``U_1 = C`` and ``U_2 = I_q``.
    """
    if not np.allclose(Q, Q.flat[0] * np.eye(len(Q)) if Q.ndim == 2 else Q.flat[0]):
        raise ValueError("The surrogate needs a state cost that is a multiple of the identity.")
    C = model.decode.T
    Q_z = Q.flat[0] * C.T @ C

    # Initial state covariance equal to the identity
    box = np.sqrt(3.0) * np.ones(model.n_hat)

    return _IdentifiedLQR(settings=LQRDiscreteSettings(
        model=StateSpaceModel(A=model.A, B=model.B),
        Q=Q_z,
        R=R,
        max_initial_state=box,
        resample_initial_state=True,
        noise_settings=None,
        seed_seq=seed_seq,
        dt=model.dt,
        U_1=model.decode.T,
        U_2=np.eye(model.B.shape[1]),
    ))
