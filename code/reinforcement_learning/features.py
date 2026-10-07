"""The policy architectures of chapters 5 and 6, as Stable-Baselines3 feature extractors.

References
----------
.. [1] R. Jonschkowski, R. Hafner, J. Scholz, M. Riedmiller, "PVEs:
       Position-Velocity Encoders for Unsupervised Learning of Structured State
       Representations", RSS 2017 Workshop on New Frontiers for Deep Learning
       in Robotics. https://arxiv.org/abs/1705.09805
       :class:`PosVelExtractor` and :func:`pretrain_pve`
.. [2] R. Jonschkowski, O. Brock, "Learning State Representations with Robotic
       Priors", Autonomous Robots 39(3), 2015.
       https://doi.org/10.1007/s10514-015-9459-7
       The five robotic priors.
.. [3] N. Botteghi, K. Alaa, M. Poel, B. Sirmacek, C. Brune, A. Mersha,
       S. Stramigioli, "Low Dimensional State Representation Learning with
       Robotics Priors in Continuous Action Spaces", 2021.
       https://arxiv.org/abs/2107.01667
       The continuous-action form of [2] for :func:`pretrain_state_encoder`.
.. [4] A. Todorov, M. Sabatelli, "Learning in Low-Dimensional Subspaces:
       Orthogonal Bottlenecks for Reinforcement Learning", RLC 2026.
       https://arxiv.org/abs/2605.26012
       :class:`OrthoBottleneck` and ``bottleneck_dim = 2`` for classic control.
"""

import numpy as np
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__all__ = [
    "LatentExtractor",
    "FCExtractor",
    "PosVelExtractor",
    "OrthoBottleneck",
    "StateEncoder",
    "pretrain_state_encoder",
    "fit_extractor_weights",
]


def _mlp(in_width: int, hidden: tuple) -> tuple[list[nn.Module], int]:
    """Flatten followed by ``Linear + ReLU`` layers; returns the layers and output width."""
    layers: list[nn.Module] = [nn.Flatten()]
    for size in hidden:
        layers += [nn.Linear(in_width, size), nn.ReLU()]
        in_width = size
    return layers, in_width


class LatentExtractor(BaseFeaturesExtractor):
    """Common base: input scaling and the :meth:`latent` hook.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        A flat observation ``(p,)`` or a frame stack ``(k, H, W)``.
    features_dim : int
        Width handed to the policy and value networks.
    """

    def __init__(self, observation_space, features_dim: int):
        super().__init__(observation_space, features_dim)
        shape = observation_space.shape
        self.input_width = int(np.prod(shape))
        if len(shape) == 3:
            self.n_frames, self.height, self.width = shape

    @staticmethod
    def _prepare(observations: torch.Tensor) -> torch.Tensor:
        """Float in ``[0, 1]``."""
        if observations.dtype == torch.uint8:
            return observations.float() / 255.0
        return observations.float()

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _register_latent_scaling(self, latent_dim: int) -> None:
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_scale", torch.ones(latent_dim))

    def latent(self, observations: torch.Tensor) -> torch.Tensor:
        """The architecture's intermediate state, of shape ``(B, d)``."""
        z = self._raw_latent(observations)
        if hasattr(self, "latent_mean"):
            z = (z - self.latent_mean) / self.latent_scale
        return z

    @torch.no_grad()
    def calibrate_latent(self, observations: torch.Tensor) -> None:
        """Fix the latent's mean and scale from the fitting data."""
        if not hasattr(self, "latent_mean"):
            return
        z = self._raw_latent(observations)
        scale = z.std(dim=0)
        scale[scale < 1e-6] = 1.0
        self.latent_mean.copy_(z.mean(dim=0))
        self.latent_scale.copy_(scale)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Features for the policy and value heads."""
        return self.head(self.latent(observations))


class FCExtractor(LatentExtractor):
    """Fully connected feature extractor that flattens the input.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        Frame stack.
    features_dim : int, optional
        Output width.
    hidden : tuple, optional
        Hidden layer widths.
    """

    def __init__(self, observation_space, features_dim: int = 256, hidden: tuple = (512, 256)):
        super().__init__(observation_space, features_dim)

        layers, width = _mlp(self.input_width, hidden)
        self.net = nn.Sequential(*layers, nn.Linear(width, features_dim), nn.ReLU())
        self.head = nn.Identity()

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """The final layer."""
        return self.net(self._prepare(observations))


class PosVelExtractor(LatentExtractor):
    """Position state encoded per frame, and velocity as its scaled frame difference.

    Follows Jonschkowski et al. [1]_, Sec. III-A and the network of Sec. III-C.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        Frame stack of at least two frames.
    position_dim : int, optional
        Width of the position state, five in [1]_.
    alpha : float, optional
        Velocity scaling, which also subsumes the time step.
    """

    def __init__(self, observation_space, position_dim: int = 5, alpha: float = 1.0):
        super().__init__(observation_space, 2 * position_dim)

        if self.n_frames < 2:
            raise ValueError("A position-velocity encoder needs at least two frames.")

        self.position_dim = position_dim

        # Shared across frames
        layers: list[nn.Module] = []
        channels, h, w = 1, self.height, self.width
        for out in (16, 32, 64):
            layers += [nn.Conv2d(channels, out, 5, stride=2, padding=2), nn.ReLU()]
            channels, h, w = out, (h + 1) // 2, (w + 1) // 2
        hidden, width = _mlp(channels * h * w, (128, 128))
        self.encoder = nn.Sequential(*layers, *hidden, nn.Linear(width, position_dim))

        # Buffer
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        self.head = nn.Identity()

    def positions(self, frames: torch.Tensor) -> torch.Tensor:
        """Position states ``(B, k, position_dim)`` of scaled frames ``(B, k, H, W)``."""
        batch, k = frames.shape[:2]
        flat = frames.reshape(batch * k, 1, self.height, self.width)
        return self.encoder(flat).view(batch, k, self.position_dim)

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """Newest position state, stacked with the velocity state."""
        s = self.positions(self._prepare(observations))
        return torch.cat([s[:, -1], self.alpha * (s[:, -1] - s[:, -2])], dim=1)


class OrthoBottleneck(LatentExtractor):
    """A learned encoder read through a fixed orthonormal projection.

    Follows Todorov and Sabatelli [4]_.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        A flat observation or a frame stack.
    bottleneck_dim : int, optional
        Width of the code, ``k``. Two suffices on classic control in [4]_, and
        is the dimension of the true state here.
    encoder_dim : int, optional
        Width of the trunk the projection compresses, ``D``.
    hidden : tuple, optional
        Hidden layer widths of the trunk.
    seed : int, optional
        Draws the projection.
    """

    def __init__(self, observation_space, bottleneck_dim: int = 2, encoder_dim: int = 256,
                 hidden: tuple = (512, 256), seed: int = 0):
        super().__init__(observation_space, bottleneck_dim)

        if bottleneck_dim > encoder_dim:
            raise ValueError("A bottleneck wider than the encoder compresses nothing.")

        layers, width = _mlp(self.input_width, hidden)
        self.net = nn.Sequential(*layers, nn.Linear(width, encoder_dim), nn.ReLU())

        B, _ = np.linalg.qr(
            np.random.default_rng(seed).standard_normal((encoder_dim, bottleneck_dim))
        )
        # Projection matrix as a buffer
        self.register_buffer("B", torch.as_tensor(B, dtype=torch.float32))

        self.head = nn.Identity()

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """The projected code. Left unscaled, as ``B`` is already an isometry."""
        return self.net(self._prepare(observations)) @ self.B


class StateEncoder(LatentExtractor):
    """Learned low-dimensional state.

    Uses the robotic priors of [2]_ and [3]_.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        A flat observation or a frame stack.
    latent_dim : int, optional
        Width of the state estimate.
    hidden : tuple, optional
        Hidden layer widths.
    features_dim : int, optional
        Output width.
    """

    def __init__(self, observation_space, latent_dim: int = 2, hidden: tuple = (512, 256),
                 features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        layers, width = _mlp(self.input_width, hidden)
        self.net = nn.Sequential(*layers, nn.Linear(width, latent_dim))

        self.head = nn.Sequential(nn.Linear(latent_dim, features_dim), nn.ReLU())
        self._register_latent_scaling(latent_dim)

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """The state estimate."""
        return self.net(self._prepare(observations))


def _collect(env, policy, n_steps: int):
    """Roll out ``policy`` for observations.

    Parameters
    ----------
    env : gymnasium.Env
        Environment to drive. Restored afterwards if it can be.
    policy : callable
        ``policy(observation, info) -> action``, the exploration policy.
    n_steps : int
        Steps to collect.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Observations ``(n_steps, ...)``, the actions that produced them
        ``(n_steps, q)``, and a boolean mask marking the steps whose successor
        belongs to the same episode.
    """
    snap = env.snapshot() if hasattr(env, "snapshot") else None

    observations = np.empty((n_steps, *env.observation_space.shape),
                            dtype=env.observation_space.dtype)
    actions = np.empty((n_steps, *env.action_space.shape), dtype=np.float32)
    linked = np.ones(n_steps, dtype=bool)
    linked[-1] = False

    try:
        observation, info = env.reset()
        for step in range(n_steps):
            action = np.asarray(policy(observation, info), dtype=np.float32)
            observation, _, terminated, truncated, info = env.step(action)

            observations[step] = observation
            actions[step] = action

            if terminated or truncated:
                linked[step] = False
                observation, info = env.reset()
    finally:
        if snap is not None:
            env.restore(snap)

    return observations, actions, linked


def _fit(extractor: LatentExtractor, data: torch.Tensor, n: int, loss_fn,
         epochs: int, batch_size: int = 128, learning_rate: float = 1e-3) -> dict:
    """Minibatch Adam over indices ``0..n-1``, then calibrate; returns the ``state_dict``."""
    extractor.train()
    optimizer = torch.optim.Adam(extractor.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(0)

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            loss = loss_fn(order[start:start + batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    extractor.eval()
    extractor.calibrate_latent(data)

    return {k: v.detach().cpu().clone() for k, v in extractor.state_dict().items()}


def pretrain_pve(extractor: PosVelExtractor, observations: np.ndarray, linked: np.ndarray,
                 weights: dict, epochs: int = 20, sequence_length: int = 10,
                 batch_sequences: int = 32, jitter: float = 1e-3) -> dict:
    """Fit a position-velocity encoder from unsupervised robotic priors.

    The priors, the minibatches of short sequences and the velocity scaling
    curriculum of Jonschkowski et al. [1]_, Sec. III-B and III-C.

    Parameters
    ----------
    extractor : PosVelExtractor
        The architecture to fit, modified in place. Its ``alpha`` is the final
        velocity scaling, and is left unchanged.
    observations, linked : np.ndarray
        As returned by :func:`_collect`.
    weights : dict
        Loss weights, keyed ``variation``, ``slowness``, ``inertia``,
        ``inertia_abs`` and ``conservation``.
    epochs : int, optional
        Passes over the data, the first half with ``alpha = 0`` and the second
        with ``alpha`` ramped linearly to its final value.
    sequence_length : int, optional
        Steps per sequence.
    batch_sequences : int, optional
        Sequences per minibatch.
    jitter : float, optional
        Gaussian noise on the encoded positions, so that the norms have a
        gradient at zero ([1]_, footnote 3).

    Returns
    -------
    dict
        The fitted ``state_dict``.
    """
    data = torch.from_numpy(observations)

    # Sequence starts whose steps all belong to one episode
    breaks = np.concatenate([[0], np.cumsum(~linked)])
    starts = [
        i for i in range(0, len(linked) - sequence_length + 1, sequence_length)
        if breaks[i + sequence_length - 1] == breaks[i]
    ]
    sequences = torch.as_tensor(starts)[:, None] + torch.arange(sequence_length)

    alpha_final = float(extractor.alpha)
    phase = epochs // 2
    batches = -(-len(starts) // batch_sequences)
    ramp = max((epochs - phase) * batches, 1)
    step = 0

    def loss_fn(index):
        nonlocal step
        alpha = alpha_final * min(max(step - phase * batches + 1, 0) / ramp, 1.0)
        step += 1

        frames = extractor._prepare(data[sequences[index]])
        n, length = frames.shape[:2]

        positions = extractor.positions(
            torch.cat(
                [frames[:, 0, :1], frames[:, :, -1]], dim=1
            ).flatten(0, 1)[:, None]
        ).view(n, length + 1, -1)
        
        positions = positions + jitter * torch.randn_like(positions)

        velocity = alpha * positions.diff(dim=1)
        acceleration = velocity.diff(dim=1)
        speed = velocity.norm(dim=2)

        # All pairs at the same time step, from different sequences
        pairs = torch.triu_indices(n, n, offset=1)
        distance = (positions[pairs[0]] - positions[pairs[1]]).norm(dim=2)

        terms = {
            "variation": torch.exp(-distance).mean(),
            "slowness": positions.diff(dim=1).pow(2).sum(2).mean(),
            "inertia": acceleration.pow(2).sum(2).mean(),
            "inertia_abs": acceleration.norm(dim=2).mean(),
            "conservation": speed.diff(dim=1).pow(2).mean(),
        }
        return sum(weights[name] * value for name, value in terms.items())

    return _fit(extractor, data, len(starts), loss_fn, epochs, batch_size=batch_sequences)


def pretrain_state_encoder(extractor: StateEncoder, observations: np.ndarray,
                           actions: np.ndarray, linked: np.ndarray, epochs: int = 20,
                           alpha: float = 1.0, beta: float = 1.0,
                           weights: tuple = (1.0, 1.0, 1.0, 2.0)) -> dict:
    """Fit a state encoder from robotic priors, without labels.

    The priors are Jonschkowski and Brock's [2]_ in the continuous-action form
    of Botteghi et al. [3]_.

    Parameters
    ----------
    extractor : StateEncoder
        The architecture to fit, modified in place.
    observations, actions, linked : np.ndarray
        As returned by :func:`_collect`.
    epochs : int, optional
        Passes over the data.
    alpha : float, optional
        Weights the action magnitude in the temporal coherence prior.
    beta : float, optional
        Weights the action difference in the three pairwise priors.
    weights : tuple, optional
        ``(temporal, proportionality, repeatability, causality)``. The
        ``omega`` of [3]_, whose L2 term is left to the optimizer instead.

    Returns
    -------
    dict
        The fitted ``state_dict``.
    """
    data = torch.from_numpy(observations)
    controls = torch.from_numpy(actions)
    usable = torch.from_numpy(np.flatnonzero(linked))
    w_temporal, w_proportion, w_repeat, w_causal = weights

    def loss_fn(index):
        first = usable[index]
        second = usable[torch.randperm(usable.shape[0])[:first.shape[0]]]

        s_1, s_2 = extractor.latent(data[first]), extractor.latent(data[second])
        d_1 = extractor.latent(data[first + 1]) - s_1
        d_2 = extractor.latent(data[second + 1]) - s_2

        same_action = torch.exp(-beta * (controls[first] - controls[second]).pow(2).sum(1))
        same_state = torch.exp(-(s_2 - s_1).pow(2).sum(1))

        temporal = (d_1.pow(2).sum(1) * torch.exp(-alpha * controls[first].norm(dim=1))).mean()
        proportion = ((d_2.norm(dim=1) - d_1.norm(dim=1)).pow(2) * same_action).mean()
        repeatability = ((d_2 - d_1).pow(2).sum(1) * same_state * same_action).mean()
        causality = (same_state * same_action).mean()

        return (w_temporal * temporal + w_proportion * proportion
                + w_repeat * repeatability + w_causal * causality)

    return _fit(extractor, data, usable.shape[0], loss_fn, epochs)


def fit_extractor_weights(preset, env, explore, n_steps: int = 20_000,
                          epochs: int = 20) -> dict | None:
    """Pretrain a preset's extractor.

    Parameters
    ----------
    preset : RLSolverSettings
        The solver, read for its extractor class and kwargs so the fitted
        module is identical to the one the policy will be built with.
    env : gymnasium.Env
        Environment experience is collected from.
    explore : callable
        ``explore(observation, info) -> action``, the exploration policy.
    n_steps, epochs : int, optional
        Steps to collect and fitting epochs.

    Returns
    -------
    dict or None
        The fitted ``state_dict``, or ``None`` for an architecture that is
        learned by the RL algorithm instead.
    """
    extractor_class = preset.features_extractor_class

    if extractor_class not in (PosVelExtractor, StateEncoder):
        return None

    extractor = extractor_class(env.observation_space, **preset.features_extractor_kwargs)
    observations, actions, linked = _collect(env, explore, n_steps)

    if extractor_class is PosVelExtractor:
        import settings

        return pretrain_pve(extractor, observations, linked, settings.PVE_LOSS_WEIGHTS, epochs)

    return pretrain_state_encoder(extractor, observations, actions, linked, epochs)
