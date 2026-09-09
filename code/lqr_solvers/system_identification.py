"""Subspace identification of an unknown plant, for the model-based solvers.

The model-based solvers in :mod:`lqr_solvers.mb_policy_gradient` and
:mod:`lqr_solvers.structured_policy_iteration` read ``A``, ``B``, ``Q`` and ``R``
from the plant. This module produces those four matrices from *data*
instead, so the same solvers can run as a model-free surrogate.

References
----------
.. [1] P. Van Overschee, B. De Moor, "N4SID: Subspace algorithms for the
       identification of combined deterministic-stochastic systems",
       Automatica 30(1), 1994.
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
    "to_plant",
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
    input_rank_tol : float, optional
        Relative singular-value cutoff deciding ``m_hat = rank(B)``.
    """

    trajectory_length: int
    excitation: float
    seed_seq: np.random.SeedSequence
    max_order: int = 10
    horizon: int = 10
    order_threshold: float = 1e-3
    fixed_order: int | None = None
    input_rank_tol: float = 1e-3


@dataclass
class IdentifiedModel:
    """A plant reconstructed from data, in its own coordinates.

    The identified system is ``z' = A z + B w`` with stage cost
    ``z' Q z + w' R w``, where ``z`` is the identified state and ``w`` the
    reduced action.

    Attributes
    ----------
    A : np.ndarray
        Identified state matrix of shape ``(n_hat, n_hat)``
    B : np.ndarray
        Identified input matrix of shape ``(n_hat, m_hat)``
    Q : np.ndarray
        State cost of shape ``(n_hat, n_hat)``, regressed from observed costs.
    R : np.ndarray
        Control cost of shape ``(m_hat, m_hat)``, regressed alongside ``Q``.
    decode : np.ndarray
        ``(n_hat, p)``, mapping an observation to the identified state.
    encode : np.ndarray
        ``(q, m_hat)``, mapping a reduced action back to the controller's space.
    n_hat, m_hat : int
        Estimated state and input dimension.
    cost_residual : float
        Relative residual of the cost regression.
    effective_max_order : int
        The order bound that was enforced, see :func:`effective_max_order`.
    dt : float
        Sampling interval the model was identified at.
    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    decode: np.ndarray
    encode: np.ndarray
    n_hat: int
    m_hat: int
    cost_residual: float
    effective_max_order: int
    dt: float

    def to_controller_gain(self, K: np.ndarray) -> np.ndarray:
        """Map a gain on the identified model back to the controller's space.

        Parameters
        ----------
        K : np.ndarray
            Gain of shape ``(m_hat, n_hat)``, applied as ``w = -K z``.

        Returns
        -------
        np.ndarray
            Gain of shape ``(q, p)``, applied as ``v = -K y``.
        """
        return self.encode @ K @ self.decode

    def to_identified_gain(self, K: np.ndarray) -> np.ndarray:
        """Map a gain in the controller's space onto the identified model.

        Parameters
        ----------
        K : np.ndarray
            Gain of shape ``(q, p)``, applied as ``v = -K y``.

        Returns
        -------
        np.ndarray
            Gain of shape ``(m_hat, n_hat)``.
        """
        return self.encode.T @ K @ np.linalg.pinv(self.decode)


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
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(Y, V, costs)`` with ``Y`` of shape ``(p, T)``, ``V`` of shape
        ``(q, T)`` and ``costs`` of shape ``(T,)``.
    """
    rng = np.random.default_rng(settings.seed_seq)

    plant.reset()

    observations = []
    actions = []
    costs = []

    for _ in range(settings.trajectory_length):
        y = plant.observe()
        v = -K_0 @ y + settings.excitation * rng.standard_normal(plant.action_dim)

        step = plant.apply_control(v)

        observations.append(step.y_t)
        actions.append(step.v_t)
        costs.append(step.cost)

    return (
        np.asarray(observations).T,
        np.asarray(actions).T,
        np.asarray(costs, dtype=float),
    )


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


def _reduce_input_space(B: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Restrict the action space of the identified system to only values that affect the plant.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(B_reduced, encode, m_hat)``.
    """
    _, singular_values, Vt = np.linalg.svd(B, full_matrices=False)

    m_hat = max(1, int(np.sum(singular_values > tol * singular_values[0])))
    encode = Vt[:m_hat].T

    return B @ encode, encode, m_hat


def effective_max_order(settings: IdentificationSettings) -> int:
    """The order bound actually in force, might have been lowered due to the Hankel horizon.
    """
    if settings.fixed_order is not None:
        return int(settings.fixed_order)

    return min(int(settings.max_order), int(settings.horizon))


def _symmetric_basis(dim: int) -> list[np.ndarray]:
    """Basis of the symmetric ``dim x dim`` matrices, as used by the cost fit."""
    basis = []
    for i in range(dim):
        for j in range(i, dim):
            element = np.zeros((dim, dim))
            element[i, j] = element[j, i] = 1.0
            basis.append(element)
    return basis


def _project_psd(M: np.ndarray) -> np.ndarray:
    """Nearest positive semi-definite matrix, by clipping the eigenvalues."""
    eigenvalues, eigenvectors = np.linalg.eigh((M + M.T) / 2.0)
    return eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T


def _fit_cost_weights(Z, W, costs):
    """Recover the cost weights from the stage costs we observed

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(Q, R, relative_residual)``, both weights projected onto the PSD cone.
    """
    state_basis = _symmetric_basis(Z.shape[0])
    action_basis = _symmetric_basis(W.shape[0])

    columns = [np.einsum("it,ij,jt->t", Z, element, Z) for element in state_basis]
    columns += [np.einsum("it,ij,jt->t", W, element, W) for element in action_basis]
    design = np.asarray(columns).T

    theta, *_ = np.linalg.lstsq(design, costs, rcond=None)

    scale = np.linalg.norm(costs)
    residual = float(np.linalg.norm(design @ theta - costs) / scale) if scale > 0 else 0.0

    split = len(state_basis)
    Q = sum(c * element for c, element in zip(theta[:split], state_basis))
    R = sum(c * element for c, element in zip(theta[split:], action_basis))

    return _project_psd(Q), _project_psd(R), residual


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
    Y, V, costs = collect_identification_data(plant, K_0, settings)

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

    B, encode, m_hat = _reduce_input_space(B, settings.input_rank_tol)

    # C is orthonormal after normalization, so its transpose is its
    # pseudo-inverse
    decode = C.T

    Q, R, residual = _fit_cost_weights(decode @ Y, encode.T @ V, costs)

    return IdentifiedModel(
        A=A,
        B=B,
        Q=Q,
        R=R,
        decode=decode,
        encode=encode,
        n_hat=n_hat,
        m_hat=m_hat,
        cost_residual=residual,
        effective_max_order=bound,
        dt=plant.dt,
    )


class _IdentifiedLQR(LQRDiscrete):
    """A plant whose matrices are already discrete, identified by N4SID.
    """

    def __init__(self, settings: LQRDiscreteSettings):
        LQRBase.__init__(self, settings)
        self.dt = settings.dt


def to_plant(model: IdentifiedModel, seed_seq: np.random.SeedSequence) -> LQRDiscrete:
    """Wrap an identified model as a plant the model-based solvers can consume.

    The result is a surrogate: the solvers optimize against it exactly as they
    would against the real system, and the gain that comes out is in the
    identified coordinates. Map it back with
    :meth:`IdentifiedModel.to_controller_gain` before scoring it.

    Parameters
    ----------
    model : IdentifiedModel
        The identified system.
    seed_seq : np.random.SeedSequence
        Seeds the surrogate's initial state.

    Returns
    -------
    LQRDiscrete
        A plain, uninflated system on ``(A, B, Q, R)`` of the identified mode
    """
    # Initial state covariance equal to the identity
    box = np.sqrt(3.0) * np.ones(model.n_hat)

    return _IdentifiedLQR(settings=LQRDiscreteSettings(
        model=StateSpaceModel(A=model.A, B=model.B),
        Q=model.Q,
        R=model.R,
        max_initial_state=box,
        resample_initial_state=True,
        noise_settings=None,
        seed_seq=seed_seq,
        dt=model.dt,
    ))
