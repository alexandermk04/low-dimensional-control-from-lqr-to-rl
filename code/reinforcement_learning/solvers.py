"""The reinforcement-learning solvers, with different policy architectures.

Every solver of chapters 5 and 6 is PPO [1]_ through Stable-Baselines3 [2]_.

References
----------
.. [1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal
       Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347
.. [2] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann,
       "Stable-Baselines3: Reliable Reinforcement Learning Implementations",
       JMLR 22(268), 2021. https://github.com/DLR-RM/stable-baselines3
"""

from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.metrics import Policy

__all__ = [
    "RLSolverSettings",
    "build_model",
    "extract_linear_gain",
    "set_linear_gain",
    "freeze_actor_bias",
    "is_linear_solver",
    "load_extractor",
    "SB3Policy",
    "vector_presets",
    "pixel_presets",
]


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
        Initial optimizer step size.
    anneal_timesteps : int or None
        Environment steps over which the learning rate decays. ``None`` uses
        ``total_timesteps``.
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
    policy : str
        Stable-Baselines3 policy class name. ``"MlpPolicy"`` for the chapter-5
        solvers; ``"CnnPolicy"`` for the chapter-6 ones.
    features_extractor_class : type or None
        Custom feature extractor class, from
        :mod:`reinforcement_learning.features`. ``None`` uses the default for
        the policy.
    features_extractor_kwargs : dict
        Forwarded to the extractor's constructor.
    normalize_images : bool
        Default is ``True``.
    pretrained_extractor : dict or None
        ``state_dict`` to load into the extractor before training.
    freeze_extractor : bool
        Stop the extractor's parameters being trained, if one is passed.
    linear : bool or None
        Overrides :func:`is_linear_solver`.
    n_steps, batch_size, n_epochs, ent_coef, clip_range, max_grad_norm : ...
        PPO hyperparameters. Defaults to Stable-Baselines3 settings.
    target_kl : float or None
        Ends an update's epochs early once the approximate KL divergence to the
        old policy exceeds ``1.5 * target_kl``.
    device : str
        Torch device. ``"auto"`` follows Stable-Baselines3.
    """

    name: str
    algo: str = "ppo"
    net_arch: list | dict | None = None
    total_timesteps: int = 300_000
    gamma: float = 0.999
    learning_rate: float = 3e-4
    anneal_timesteps: int | None = None
    n_envs: int = 1
    eval_freq: int = 10_000
    log_std_init: float = -2.0
    initial_gain: np.ndarray | None = None
    freeze_bias: bool = True
    policy_kwargs_extra: dict = field(default_factory=dict)

    policy: str = "MlpPolicy"
    features_extractor_class: type | None = None
    features_extractor_kwargs: dict = field(default_factory=dict)
    normalize_images: bool = True
    pretrained_extractor: dict | None = None
    freeze_extractor: bool = False
    linear: bool | None = None

    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.0
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    device: str = "auto"


# Extractable gain matrix
LINEAR_PPO = RLSolverSettings(
    name="ppo_linear",
    algo="ppo",
    net_arch=dict(pi=[], vf=[64, 64]),
)

MLP_PPO = RLSolverSettings(
    name="ppo_mlp",
    algo="ppo",
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)

# Sees an identified, low-dimensional state, acts in the full output
ENCODER_PPO = RLSolverSettings(
    name="ppo_encoder",
    algo="ppo",
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)


def _extractor_solver(name: str, extractor, **kwargs) -> RLSolverSettings:
    """Preset for a solver whose policy observes through an extractor.

    Parameters
    ----------
    name : str
        Solver label.
    extractor : type
        The feature extractor class.
    **kwargs
        Overrides.

    Returns
    -------
    RLSolverSettings
        The preset.
    """
    defaults = dict(
        algo="ppo",
        policy="MlpPolicy",
        features_extractor_class=extractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        linear=False,
    )
    defaults.update(kwargs)

    return RLSolverSettings(name=name, **defaults)


def vector_presets() -> dict:
    """Solvers for chapter 5

    Returns
    -------
    dict
        Mapping solver name to :class:`RLSolverSettings`.
    """
    import settings
    from reinforcement_learning.features import OrthoBottleneck, StateEncoder

    return {
        "ppo_linear": LINEAR_PPO,
        "ppo_mlp": MLP_PPO,
        "ppo_encoder": ENCODER_PPO,
        "ppo_bottleneck": _extractor_solver(
            "ppo_bottleneck", OrthoBottleneck,
            features_extractor_kwargs={"bottleneck_dim": settings.BOTTLENECK_DIM},
        ),
        "ppo_encoder_scalable": _extractor_solver(
            "ppo_encoder_scalable", StateEncoder,
            features_extractor_kwargs={"latent_dim": settings.ENCODER_LATENT_DIM},
            freeze_extractor=True,
        ),
    }


# --- Chapter 6 -----------------------------------------

def pixel_gamma(horizon_seconds: float = 10.0) -> float:
    """Discount that holds the effective horizon fixed in *seconds*.

    Parameters
    ----------
    horizon_seconds : float, optional
        Effective horizon to hold fixed. Defaults to ``10.0``, chapter 5's.

    Returns
    -------
    float
        The discount to use at the chapter-6 agent rate.
    """
    from config import DEFAULT_ACTION_REPEAT, DEFAULT_DT

    return 1.0 - (DEFAULT_DT * DEFAULT_ACTION_REPEAT) / horizon_seconds


def _pixel_solver(name: str, extractor, **kwargs) -> RLSolverSettings:
    """Chapter-6 preset, with shared settings.

    Parameters
    ----------
    name : str
        Solver label.
    extractor : type
        The feature extractor class.
    **kwargs
        Overrides.

    Returns
    -------
    RLSolverSettings
        The preset.
    """
    defaults = dict(
        algo="ppo",
        policy="CnnPolicy",
        features_extractor_class=extractor,
        net_arch=dict(pi=[64], vf=[64]),
        linear=False,
        freeze_bias=False,
        gamma=pixel_gamma(),
        learning_rate=3e-4,
        log_std_init=-1.0,
        total_timesteps=150_000,
        eval_freq=10_000,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        device="cpu",
    )
    defaults.update(kwargs)

    return RLSolverSettings(name=name, **defaults)


def pixel_presets() -> dict:
    """Solvers for chapter 6

    Returns
    -------
    dict
        Mapping solver name to :class:`RLSolverSettings`.
    """
    import config as cfg
    import settings
    from reinforcement_learning.features import (
        FCExtractor,
        OrthoBottleneck,
        PosVelExtractor,
        StateEncoder,
    )

    dt_agent = cfg.DEFAULT_DT * cfg.DEFAULT_ACTION_REPEAT

    return {
        "ppo_pixel_fc": _pixel_solver("ppo_pixel_fc", FCExtractor),
        "ppo_pixel_posvel": _pixel_solver(
            "ppo_pixel_posvel", PosVelExtractor,
            features_extractor_kwargs={
                "position_dim": settings.PVE_POSITION_DIM, "alpha": 1.0 / dt_agent,
            },
            freeze_extractor=True,
        ),
        "ppo_pixel_ortho": _pixel_solver(
            "ppo_pixel_ortho", OrthoBottleneck,
            features_extractor_kwargs={"bottleneck_dim": settings.BOTTLENECK_DIM},
        ),
        "ppo_pixel_estimator": _pixel_solver(
            "ppo_pixel_estimator", StateEncoder,
            features_extractor_kwargs={"latent_dim": settings.ENCODER_LATENT_DIM},
            freeze_extractor=True,
        ),
    }


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
    if settings.linear is not None:
        return settings.linear

    if settings.algo != "ppo":
        return False

    arch = settings.net_arch
    if isinstance(arch, dict):
        return len(arch.get("pi", [1])) == 0

    return arch is not None and len(arch) == 0


def build_model(env, settings: RLSolverSettings, seed: int) -> BaseAlgorithm:
    """Construct the SB3 model for one solver.

    Follows the standard PPO practice of [3]_ and [4]_.

    References
    ----------
    .. [3] M. Andrychowicz et al., "What Matters In On-Policy Reinforcement
           Learning? A Large-Scale Empirical Study", ICLR 2021.
           https://arxiv.org/abs/2006.05990
    .. [4] S. Huang, R. F. J. Dossa, A. Raffin, A. Kanervisto, W. Wang, "The 37
           Implementation Details of Proximal Policy Optimization", ICLR Blog
           Track 2022.
           https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

    Parameters
    ----------
    env : gymnasium.Env
        The training environment, e.g. an
        :class:`~reinforcement_learning.lqr_env.LQREnv`.
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
    if settings.algo != "ppo":
        raise ValueError(
            f"Unknown algorithm {settings.algo!r}; expected 'ppo'."
        )

    policy_kwargs = {"net_arch": settings.net_arch, **settings.policy_kwargs_extra}

    policy_kwargs.setdefault("ortho_init", False)
    policy_kwargs.setdefault("log_std_init", settings.log_std_init)

    if settings.features_extractor_class is not None:
        policy_kwargs["features_extractor_class"] = settings.features_extractor_class
        policy_kwargs["features_extractor_kwargs"] = dict(settings.features_extractor_kwargs)
        policy_kwargs["normalize_images"] = settings.normalize_images

        if not settings.freeze_extractor:
            policy_kwargs.setdefault("share_features_extractor", False)

    vec_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(env)]),
        norm_obs=False, norm_reward=True, gamma=settings.gamma,
    )

    total = settings.total_timesteps
    horizon = settings.anneal_timesteps or total

    def learning_rate(progress_remaining: float) -> float:
        return settings.learning_rate * max(1.0 - (1.0 - progress_remaining) * total / horizon, 0.0)

    algorithm_kwargs = {
        "n_steps": settings.n_steps,
        "batch_size": settings.batch_size,
        "n_epochs": settings.n_epochs,
        "ent_coef": settings.ent_coef,
        "clip_range": settings.clip_range,
        "max_grad_norm": settings.max_grad_norm,
        "target_kl": settings.target_kl,
    }

    model = PPO(
        settings.policy,
        vec_env,
        seed=seed,
        gamma=settings.gamma,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        device=settings.device,
        verbose=0,
        **algorithm_kwargs,
    )

    if settings.pretrained_extractor is not None:
        load_extractor(model, settings.pretrained_extractor, freeze=settings.freeze_extractor)
    elif settings.freeze_extractor:
        raise ValueError("freeze_extractor=True without a pretrained_extractor freezes noise.")

    if settings.initial_gain is not None:
        set_linear_gain(model, settings.initial_gain)

    if settings.freeze_bias and is_linear_solver(settings):
        freeze_actor_bias(model)

    return model


def load_extractor(model: BaseAlgorithm, state_dict: dict, freeze: bool = True) -> None:
    """Reload pre-fitted weights into a policy's feature extractor.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        The model whose extractor should be replaced.
    state_dict : dict
        Weights, as produced by the pretraining routines in
        :mod:`reinforcement_learning.features`.
    freeze : bool, optional
        Also stop the extractor being trained. Defaults to ``True``.

    Returns
    -------
    None

    """
    extractor = model.policy.features_extractor
    extractor.load_state_dict(state_dict)

    if freeze:
        for parameter in extractor.parameters():
            parameter.requires_grad_(False)


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
    encoder : np.ndarray or None, optional
        ``(k, p)`` matrix applied to the observation before the model sees it.
    obs_scale : float, optional
        Multiplies the observation before the encoder, matching
        :class:`~reinforcement_learning.lqr_env.LQREnv`. Defaults to ``1.0``.
    """

    def __init__(self, model: BaseAlgorithm, deterministic: bool = True,
                 encoder: np.ndarray | None = None, obs_scale: float = 1.0):
        self.model = model
        self.deterministic = deterministic
        self.encoder = None if encoder is None else np.asarray(encoder, dtype=np.float64)
        self.obs_scale = float(obs_scale)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Return the control for one observation."""
        y = self.obs_scale * np.asarray(y, dtype=np.float64)
        if self.encoder is not None:
            y = self.encoder @ y

        action, _ = self.model.predict(
            np.asarray(y, dtype=np.float32), deterministic=self.deterministic
        )
        return np.asarray(action, dtype=np.float64).reshape(-1)
