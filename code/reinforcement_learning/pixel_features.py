"""Chapter 6 policy architectures, as Stable-Baselines3 feature extractors.

The position-velocity encoder (Jonschkowski, Hoefer and Brock,
arXiv:1705.09805) is fitted first by :func:`fit_extractor_weights`
and then frozen.
"""

import numpy as np
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__all__ = [
    "FCExtractor",
    "OrthoBottleneckExtractor",
    "StateEstimatorExtractor",
    "PosVelExtractor",
    "fit_extractor_weights",
]


def _mlp(in_width: int, hidden: tuple) -> tuple[list[nn.Module], int]:
    """Flatten followed by ``Linear + ReLU`` layers; returns the layers and output width."""
    layers: list[nn.Module] = [nn.Flatten()]
    for size in hidden:
        layers += [nn.Linear(in_width, size), nn.ReLU()]
        in_width = size
    return layers, in_width


class PixelExtractor(BaseFeaturesExtractor):
    """Common base: input scaling and the :meth:`latent` hook.
    """

    def __init__(self, observation_space, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.n_frames, self.height, self.width = observation_space.shape

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
        """The architecture's interpretable intermediate, of shape ``(B, d)``."""
        z = self._raw_latent(observations)
        if hasattr(self, "latent_mean"):
            z = (z - self.latent_mean) / self.latent_scale
        return z

    @torch.no_grad()
    def calibrate_latent(self, observations: torch.Tensor) -> None:
        """Fix the latent's mean and scale from the fitting data."""
        z = self._raw_latent(observations)
        scale = z.std(dim=0)
        scale[scale < 1e-6] = 1.0
        self.latent_mean.copy_(z.mean(dim=0))
        self.latent_scale.copy_(scale)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Features for the policy and value heads."""
        return self.head(self.latent(observations))


class FCExtractor(PixelExtractor):
    """Flatten the frames and learn.

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

        layers, width = _mlp(self.n_frames * self.height * self.width, hidden)
        self.net = nn.Sequential(*layers, nn.Linear(width, features_dim), nn.ReLU())
        self.head = nn.Identity()

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """The final layer, this architecture having no narrower one."""
        return self.net(self._prepare(observations))


class PosVelExtractor(PixelExtractor):
    """Keypoint positions from a spatial softmax, and velocity as their frame difference.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        Frame stack of at least two frames.
    n_keypoints : int, optional
        Spatial features tracked per frame, at least two.
    temperature : float, optional
        Softmax temperature.
    dt_agent : float, optional
        Interval between two frames, scaling the difference into a velocity.
    features_dim : int, optional
        Output width.
    """

    def __init__(self, observation_space, n_keypoints: int = 4, temperature: float = 1.0,
                 dt_agent: float = 0.1, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        if self.n_frames < 2:
            raise ValueError("A position-velocity encoder needs at least two frames.")
        if n_keypoints < 2:
            raise ValueError("A single keypoint degenerates to the frame's centroid.")

        self.n_keypoints = n_keypoints
        self.temperature = temperature
        self.dt_agent = dt_agent

        # Shared across frames
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, n_keypoints, 3, padding=1),
        )
        h, w = (self.height + 1) // 2, (self.width + 1) // 2
        self.register_buffer("grid_x", torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w))
        self.register_buffer("grid_y", torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1))

        self.head = nn.Sequential(nn.Linear(4 * n_keypoints, features_dim), nn.ReLU())
        self._register_latent_scaling(4 * n_keypoints)

    def keypoints(self, frames: torch.Tensor) -> torch.Tensor:
        """Coordinates of shape ``(B, k, 2 * n_keypoints)`` for scaled frames ``(B, k, H, W)``."""
        batch, k = frames.shape[:2]

        maps = self.conv(frames.reshape(batch * k, 1, self.height, self.width))
        weights = torch.softmax(maps.flatten(2) / self.temperature, dim=-1).view_as(maps)

        x = (weights * self.grid_x).sum(dim=(2, 3))
        y = (weights * self.grid_y).sum(dim=(2, 3))

        return torch.cat([x, y], dim=1).view(batch, k, 2 * self.n_keypoints)

    def _raw_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """Newest keypoint positions, concatenated with their rate of change."""
        points = self.keypoints(self._prepare(observations))
        velocity = (points[:, -1] - points[:, -2]) / self.dt_agent
        return torch.cat([points[:, -1], velocity], dim=1)


def _collect_frames(env, policy, n_steps: int) -> np.ndarray:
    """Observations of shape ``(n_steps, k, H, W)`` in temporal order."""
    snap = env.snapshot()
    observations = np.empty((n_steps, *env.observation_space.shape), dtype=np.uint8)

    try:
        observation, info = env.reset()
        for step in range(n_steps):
            observation, _, terminated, truncated, info = env.step(policy(info))
            observations[step] = observation
            if terminated or truncated:
                observation, info = env.reset()
    finally:
        env.restore(snap)

    return observations


def _fit(extractor: PixelExtractor, data: torch.Tensor, n: int, loss_fn,
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


def pretrain_pve(extractor: PosVelExtractor, observations: np.ndarray, epochs: int = 20,
                 w_variation: float = 1.0, w_slowness: float = 1.0, w_inertia: float = 1.0) -> dict:
    """Fit a position-velocity encoder from unsupervised robotic priors.
    """
    data = torch.from_numpy(observations)

    def loss_fn(index):
        latent_t = extractor.latent(data[index])
        latent_next = extractor.latent(data[index + 1])
        half = latent_t.shape[1] // 2

        position = latent_t[:, :half]
        shuffled = position[torch.randperm(position.shape[0])]

        variation = torch.exp(-torch.norm(position - shuffled, dim=1)).mean()
        slowness = (latent_next[:, :half] - position).pow(2).mean()
        inertia = (latent_next[:, half:] - latent_t[:, half:]).pow(2).mean()

        return w_variation * variation + w_slowness * slowness + w_inertia * inertia

    return _fit(extractor, data, data.shape[0] - 2, loss_fn, epochs)


def fit_extractor_weights(solver: str, env, oracle_gain: np.ndarray, seed: int,
                          n_steps: int = 20_000, epochs: int = 20) -> dict | None:
    """Pretrain the extractor of ``ppo_pixel_posvel``.

    Parameters
    ----------
    solver : str
        Preset name.
    env : PixelLQREnv
        Environment frames are collected from.
    oracle_gain : np.ndarray
        Coarse-rate gain.
    seed : int
        Seeds the exploration mixture.
    n_steps, epochs : int, optional
        Frames to collect and fitting epochs.

    Returns
    -------
    dict or None
        The fitted ``state_dict``.
    """
    from lqr_base.thruster_ring import ThrusterOraclePolicy

    if solver != "ppo_pixel_posvel":
        return None

    oracle = ThrusterOraclePolicy(oracle_gain, env.c)
    rng = np.random.default_rng(seed)

    def mixed(info):
        if rng.random() < 0.3:
            return rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        return oracle(info["state"])

    observations = _collect_frames(env, mixed, n_steps)

    return pretrain_pve(
        PosVelExtractor(env.observation_space, n_keypoints=4, dt_agent=env.dt_agent),
        observations, epochs,
    )
