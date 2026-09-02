"""The reinforcement-learning solvers, with different policy architectures.

**Linear solver.** PPO with an empty policy network, so the actor is a single
affine layer and the deterministic policy is exactly ``v = W y + b``. The gain
``K = -W`` can then be scored by the *same closed form* the LQR solvers are
scored by, and its rank measured.

**Neural solver.** A standard MLP actor, scored by Monte-Carlo rollout only. 
"""

from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from evaluation.metrics import Policy

__all__ = [
    "RLSolverSettings",
    "LINEAR_PPO",
    "MLP_PPO",
    "MLP_SAC",
    "build_model",
    "extract_linear_gain",
    "set_linear_gain",
    "freeze_actor_bias",
    "is_linear_solver",
    "SB3Policy",
]

ALGORITHMS = {"ppo": PPO, "sac": SAC}


@dataclass
class RLSolverSettings:
    """Hyperparameters of one RL solver.

    Attributes
    ----------
    name : str
        Solver label, used as the record's ``solver`` field.
    algo : str
        ``"ppo"`` or ``"sac"``.
    net_arch : list | dict | None
        Actor/critic architecture. ``dict(pi=[], vf=[64, 64])`` gives the
        linear actor with a usable critic; a plain list gives the same shape to
        both.
    total_timesteps : int
        Environment steps to train for.
    gamma : float
        Discount. ``0.99`` at ``dt = 0.01`` is a one-second effective horizon.
    learning_rate : float
        Optimizer step size.
    n_envs : int
        Parallel environments.
    eval_freq : int
        Environment steps between learning-curve samples.
    log_std_init : float
        Initial log standard deviation of the PPO Gaussian.
        Ignored by SAC.
    initial_gain : np.ndarray or None
        Stabilizing gain ``K_0`` to initialize a linear actor with, in the
        ``v = -K y`` convention.
    freeze_bias : bool
        Hold the linear actor's bias at zero.
        Fixes the policy to the class of pure linear feedbacks, so the RL solver is
        directly comparable to the LQR solvers.
    policy_kwargs_extra : dict
        Merged into the constructed ``policy_kwargs``.
    """

    name: str
    algo: str = "ppo"
    net_arch: list | dict | None = None
    total_timesteps: int = 300_000
    gamma: float = 0.999
    learning_rate: float = 3e-4
    n_envs: int = 1
    eval_freq: int = 10_000
    log_std_init: float = -2.0
    initial_gain: np.ndarray | None = None
    freeze_bias: bool = True
    policy_kwargs_extra: dict = field(default_factory=dict)

# Extractable gain matrix
LINEAR_PPO = RLSolverSettings(
    name="ppo_linear",
    algo="ppo",
    net_arch=dict(pi=[], vf=[64, 64]),
    learning_rate=3e-4,
)

MLP_PPO = RLSolverSettings(
    name="ppo_mlp",
    algo="ppo",
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    learning_rate=3e-4,
)

MLP_SAC = RLSolverSettings(
    name="sac_mlp",
    algo="sac",
    net_arch=[256, 256],
    total_timesteps=100_000,
    learning_rate=3e-4,
)


def is_linear_solver(settings: RLSolverSettings) -> bool:
    """Whether this solver's policy admits an exact gain.

    Parameters
    ----------
    settings : RLSolverSettings
        The solver.

    Returns
    -------
    bool
        ``True`` only for PPO with an empty actor network. SAC is excluded even
        with an empty network, because its policy isnon-linear.
    """
    if settings.algo != "ppo":
        return False

    arch = settings.net_arch
    if isinstance(arch, dict):
        return len(arch.get("pi", [1])) == 0

    return arch is not None and len(arch) == 0


def build_model(env, settings: RLSolverSettings, seed: int) -> BaseAlgorithm:
    """Construct the SB3 model for one solver.

    Parameters
    ----------
    env : gymnasium.Env or stable_baselines3.common.vec_env.VecEnv
        The training environment, an :class:`~reinforcement_learning.lqr_env.LQREnv`
        or a vectorized stack of them.
    settings : RLSolverSettings
        The solver.
    seed : int
        Seed for the algorithm's own randomness.

    Returns
    -------
    stable_baselines3.common.base_class.BaseAlgorithm
        An untrained model.

    Raises
    ------
    ValueError
        If ``settings.algo`` is not a known algorithm.
    """
    if settings.algo not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {settings.algo!r}; expected one of {sorted(ALGORITHMS)}."
        )

    policy_kwargs = {"net_arch": settings.net_arch, **settings.policy_kwargs_extra}

    if settings.algo == "ppo":
        policy_kwargs.setdefault("ortho_init", False)
        policy_kwargs.setdefault("log_std_init", settings.log_std_init)

    model = ALGORITHMS[settings.algo](
        "MlpPolicy",
        env,
        seed=seed,
        gamma=settings.gamma,
        learning_rate=settings.learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    if settings.initial_gain is not None:
        set_linear_gain(model, settings.initial_gain)

    if settings.freeze_bias and is_linear_solver(settings):
        freeze_actor_bias(model)

    return model


def freeze_actor_bias(model: BaseAlgorithm) -> None:
    """Zero a linear actor's bias and stop it being trained.

    Confines the policy to ``v = -K y``.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        A PPO model with an empty actor network.
    """
    import torch

    bias = model.policy.action_net.bias
    with torch.no_grad():
        bias.zero_()
    bias.requires_grad_(False)


def set_linear_gain(model: BaseAlgorithm, K: np.ndarray) -> None:
    """Write a feedback gain into a linear actor, as its starting policy.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        A PPO model with an empty actor network.
    K : np.ndarray
        Gain of shape ``(q, p)`` in the ``v = -K y`` convention.

    Raises
    ------
    AttributeError
        If the policy has no ``action_net``.
    ValueError
        If ``K`` does not match the actor's shape.
    """
    import torch

    policy = getattr(model, "policy", None)

    if not hasattr(policy, "action_net"):
        raise AttributeError(
            f"{type(policy).__name__} has no action_net to initialize."
        )

    weight = policy.action_net.weight
    if tuple(weight.shape) != tuple(K.shape):
        raise ValueError(
            f"K has shape {K.shape}, but the actor expects {tuple(weight.shape)}."
        )

    with torch.no_grad():
        weight.copy_(torch.as_tensor(-K, dtype=weight.dtype))
        policy.action_net.bias.zero_()


def extract_linear_gain(model: BaseAlgorithm) -> tuple[np.ndarray, np.ndarray]:
    """Recover the feedback gain from a linear PPO actor.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        A PPO model built with an empty actor network.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(K, bias)`` in this repository's ``v = -K y`` convention, so
        ``K = -W`` for an actor computing ``W y + b``. ``K`` has shape
        ``(q, p)``.

    Raises
    ------
    AttributeError
        If the policy has no ``action_net``, i.e. it is not a PPO-style actor.
    ValueError
        If the actor network is not empty, so the policy is not affine.
    """
    policy = getattr(model, "policy", None)

    if not hasattr(policy, "action_net"):
        raise AttributeError(
            f"{type(policy).__name__} has no action_net; only a PPO-style actor "
            "can be reduced to a gain."
        )

    if len(list(policy.mlp_extractor.policy_net.children())) > 0:
        raise ValueError(
            "The actor network is not empty, so the policy is not affine and no "
            "exact gain exists. Score it with monte_carlo_cost instead."
        )

    weight = policy.action_net.weight.detach().cpu().numpy().astype(np.float64)
    bias = policy.action_net.bias.detach().cpu().numpy().astype(np.float64)

    return -weight, bias


class SB3Policy:
    """Adapter presenting a trained SB3 model as a :class:`~evaluation.metrics.Policy`.

    Lets :func:`~evaluation.metrics.monte_carlo_cost` score a neural policy.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        A trained model.
    deterministic : bool, optional
        Use the distribution's mean rather than a sample. Defaults to ``True``.
    """

    def __init__(self, model: BaseAlgorithm, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Return the control for one observation."""
        action, _ = self.model.predict(
            np.asarray(y, dtype=np.float32), deterministic=self.deterministic
        )
        return np.asarray(action, dtype=np.float64).reshape(-1)
