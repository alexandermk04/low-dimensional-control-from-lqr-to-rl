"""The LQR system, presented to Stable-Baselines3 as a Gymnasium environment.

This is a *wrapper*, giving the algorithms a Gymnasium environment
interface to our LQR system.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lqr_base.simple_lqr import LQRBase

__all__ = ["LQREnv", "make_lqr_env_fns"]


class LQREnv(gym.Env):
    """Gymnasium view of an :class:`~lqr_base.simple_lqr.LQRBase`.

    Parameters
    ----------
    lqr : LQRBase
        The system to drive.
    horizon : int
        Steps before truncation.
    reward_scale : float
        Multiplies the stage cost. Passing ``dt`` makes the return a Riemann sum
        of the continuous-time cost integral.
    action_limit : float
        Half-width of the action box.
    obs_limit : float, optional
        Half-width of the observation box. Defaults to ``1e4``.
    render_mode : str or None, optional
        ``"human"`` opens a pygame window displaying the environment.
    encoder : np.ndarray or None, optional
        ``(k, p)`` matrix applied to every observation, so the policy observes
        ``encoder @ y`` instead of ``y``. ``None`` means the policy receives the full observation.
    obs_scale : float, optional
        Multiplies the plant's observation before the encoder. ``sqrt(p/n)``
        undoes the per-coordinate shrinking of an orthonormal lift. Defaults to ``1.0``.

    Attributes
    ----------
    lqr : LQRBase
        The wrapped system.
    observation_dim : int
        ``lqr.observation_dim``, or the
        encoder's output dimension when one is given.
    steps : int
        Steps taken in the current episode.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        lqr: LQRBase,
        horizon: int,
        reward_scale: float,
        action_limit: float,
        obs_limit: float = 1e4,
        render_mode: str | None = None,
        encoder: np.ndarray | None = None,
        obs_scale: float = 1.0,
    ):
        super().__init__()

        self.lqr = lqr
        self.obs_scale = float(obs_scale)
        self.horizon = horizon
        self.reward_scale = reward_scale
        self.action_limit = action_limit
        self.obs_limit = obs_limit
        self.render_mode = render_mode

        self.encoder = None if encoder is None else np.asarray(encoder, dtype=np.float64)
        if self.encoder is not None and self.encoder.shape[1] != lqr.observation_dim:
            raise ValueError(
                f"encoder maps from {self.encoder.shape[1]} dimensions but the "
                f"plant observes {lqr.observation_dim}."
            )

        self.observation_dim = (
            lqr.observation_dim if self.encoder is None else self.encoder.shape[0]
        )

        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit,
            shape=(self.observation_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-action_limit, high=action_limit,
            shape=(lqr.action_dim,), dtype=np.float32,
        )

        self.steps = 0

        self.screen = None
        self.clock = None
        self.screen_dim = 500

    def _obs(self) -> np.ndarray:
        """Current observation, scaled and encoded if asked, cast to the dtype SB3 expects."""
        y = self.obs_scale * self.lqr.observe()
        if self.encoder is not None:
            y = self.encoder @ y
        return np.asarray(y, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        Parameters
        ----------
        seed : int or None, optional
            Ignored, as we handle the random state our way.
        options : dict or None, optional
            Unused, required by the Gymnasium API.

        Returns
        -------
        tuple[np.ndarray, dict]
            Initial observation and an empty info dict.
        """
        self.lqr.reset()
        self.steps = 0

        return self._obs(), {"state": self.lqr.x.copy()}

    def step(self, action):
        """Apply one control and charge the LQR stage cost.

        Parameters
        ----------
        action : array-like
            Control in the controller's own space, of shape
            ``(lqr.action_dim,)``.

        Returns
        -------
        tuple
            The Gymnasium five-tuple. ``reward`` is the negated, scaled stage
            cost; ``terminated`` fires only if the state has left the finite
            numbers; ``truncated`` is the horizon cutoff. ``info`` carries the
            unscaled ``stage_cost`` and the true ``x`` and ``u``, so a
            trajectory can be reconstructed.
        """
        v = np.asarray(action, dtype=np.float64).reshape(self.lqr.action_dim)

        v = np.clip(v, -self.action_limit, self.action_limit)

        result = self.lqr.apply_control(v)
        self.steps += 1

        reward = -self.reward_scale * result.cost

        terminated = not np.all(np.isfinite(self.lqr.x))
        truncated = self.steps >= self.horizon

        info = {
            "stage_cost": result.cost,
            "x": result.x_t,
            "state": result.x_t,
            "u": result.u_t,
        }

        observation = (
            self._obs() if not terminated
            else np.zeros(self.observation_dim, dtype=np.float32)
        )

        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """Draw the cart and its velocity.

        Returns
        -------
        None
        """
        if self.render_mode is None:
            return

        if self.lqr.A.shape[0] != 2:
            raise NotImplementedError(
                "Rendering assumes a [position, velocity] state."
            )

        import pygame

        width, height = self.screen_dim, self.screen_dim // 2

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("LQR Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        field = pygame.Surface((width, height))
        field.fill((255, 255, 255))
        line_y = height // 2

        # The track, and the origin the controller is driving towards
        span = float(self.lqr.max_initial_state[0])
        pygame.draw.line(field, (100, 100, 100), (0, line_y), (width, line_y), 2)
        pygame.draw.circle(field, (0, 150, 0), (x_to_pixel(0.0, span, width), line_y), 5)

        x, v = self.lqr.x[0], self.lqr.x[1]
        cart_x = x_to_pixel(float(x), span, width)
        pygame.draw.rect(field, (30, 90, 200), (cart_x - 10, line_y - 10, 20, 20))
        pygame.draw.line(field, (200, 40, 40), (cart_x, line_y),
                         (int(cart_x + float(v) * 20), line_y), 2)

        self.screen.blit(field, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        """Close the pygame window, if one was opened."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


def x_to_pixel(x: float, span: float, screen_width: int) -> int:
    """Map a track position to a horizontal screen coordinate.

    Parameters
    ----------
    x : float
        Position, expected in ``[-span, span]``.
    span : float
        Half-width of the track, mapped to the screen edges.
    screen_width : int
        Window width in pixels.

    Returns
    -------
    int
        Pixel column.
    """
    return int((x + span) / (2 * span) * screen_width)


def make_lqr_env_fns(build_lqr, seed_seq, n_envs: int, **env_kwargs) -> list:
    """Build independent environments for an RL run.

    Each sub-environment has its own :class:`~lqr_base.simple_lqr.LQRBase`
    with a unique seed sequence.

    Parameters
    ----------
    build_lqr : Callable[[np.random.SeedSequence], LQRBase]
        Constructs a system from a seed sequence.
    seed_seq : np.random.SeedSequence
        Parent sequence, spawned once per environment.
    n_envs : int
        Number of parallel environments.
    **env_kwargs
        Forwarded to :class:`LQREnv`.

    Returns
    -------
    list
        Thunks suitable for ``DummyVecEnv`` or ``SubprocVecEnv``.
    """
    children = seed_seq.spawn(n_envs)

    def make(child):
        return lambda: LQREnv(build_lqr(child), **env_kwargs)

    return [make(child) for child in children]
