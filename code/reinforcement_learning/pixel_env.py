"""The chapter-6 pixelated environment
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config as cfg
from lqr_base.inflated_lqr import LQRInflated, LQRInflatedSettings

from reinforcement_learning.pixel_render import SceneSpec, SceneRenderer

__all__ = ["PixelEnvConfig", "PixelLQREnv", "build_pixel_lqr", "build_pixel_env"]


@dataclass
class PixelEnvConfig:
    """Configuration of :class:`PixelLQREnv`.

    Attributes
    ----------
    horizon : int
        Steps before truncation.
    action_repeat : int
        Plant steps one command is held for.
    frame_stack : int
        Frames handed to the policy, oldest first.
    reward_scale : float
        Multiplies the accumulated stage cost. Pass the simulation ``dt``.
    scene : SceneSpec
        Geometry of the rendered frame.
    thruster_angles : np.ndarray or None
        Thruster axes, used only for rendering.
    """

    horizon: int
    action_repeat: int
    frame_stack: int
    reward_scale: float
    scene: SceneSpec
    thruster_angles: np.ndarray | None = None


class PixelLQREnv(gym.Env):
    """Gymnasium view of a thruster-ring LQR system, observed through a camera.

    Parameters
    ----------
    lqr : LQRInflated
        The system, with ``U_1 = I`` and a thruster-ring ``U_2``.
    config : PixelEnvConfig
        Timing and framing.
    render_mode : str or None, optional
        ``"human"`` opens a pygame window, ``"rgb_array"`` returns the magnified
        frame.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, lqr: LQRInflated, config: PixelEnvConfig, render_mode: str | None = None):
        super().__init__()

        if config.action_repeat < 1 or config.frame_stack < 1:
            raise ValueError("action_repeat and frame_stack must be positive.")

        self.lqr = lqr
        self.config = config
        self.render_mode = render_mode
        self.renderer = SceneRenderer(config.scene)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(config.frame_stack, config.scene.height, config.scene.width),
            dtype=np.uint8,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(lqr.action_dim,), dtype=np.float32)

        # Along-rail effectiveness of each thruster, thrust scale included.
        self.c = np.asarray(lqr.U_2[0], dtype=float)

        self.agent_steps = 0
        self.n_action_clips = 0
        self._frames = deque(maxlen=config.frame_stack)
        self._last_thrust = None
        self.screen = None
        self.clock = None

    @property
    def dt_agent(self) -> float:
        """Interval between two agent decisions, in seconds."""
        return self.lqr.dt * self.config.action_repeat

    def _observation(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def _info_state(self) -> np.ndarray:
        return np.asarray(self.lqr.x, dtype=float).copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.
        """
        self.lqr.reset()
        self.agent_steps = 0

        frame = self.renderer.frame(float(self.lqr.x[0]))
        self._frames.extend(frame.copy() for _ in range(self.config.frame_stack))

        return self._observation(), {"state": self._info_state()}

    def step(self, action):
        """Hold one command for ``action_repeat`` plant steps."""

        raw = np.asarray(action, dtype=np.float64).reshape(self.lqr.action_dim)
        v = np.clip(raw, -1.0, 1.0)
        self.n_action_clips += int(not np.array_equal(v, raw))

        cost = 0.0
        terminated = False
        for _ in range(self.config.action_repeat):
            result = self.lqr.apply_control(v)
            cost += result.cost
            if not np.all(np.isfinite(self.lqr.x)):
                terminated = True
                break

        self.agent_steps += 1
        self._last_thrust = v
        x = float(self.lqr.x[0])
        self._frames.append(self.renderer.frame(x))

        u = float(np.ravel(result.u_t)[0])
        energy = float(v @ v)

        info = {
            "stage_cost": cost,
            "state": self._info_state(),
            "u": u,
            "thrust": v.copy(),
            "thruster_energy": energy,
            "null_waste": energy - u * u / float(self.c @ self.c),
            "off_screen": not self.renderer.is_visible(x),
        }
        reward = -self.config.reward_scale * cost
        truncated = self.agent_steps >= self.config.horizon

        return self._observation(), float(reward), terminated, truncated, info

    _SNAPSHOT_ARRAYS = ("x", "x_0", "_y_cache")

    def snapshot(self) -> dict:
        """Capture everything the current state."""
        return {
            "state_rng": self.lqr.state_rng.bit_generator.state,
            "noise_rng": self.lqr.noise_rng.bit_generator.state,
            "env_steps": self.lqr.env_steps,
            "episodes": self.lqr.episodes,
            "initialized": getattr(self.lqr, "initialized", False),
            **{
                name: None if getattr(self.lqr, name, None) is None
                else np.array(getattr(self.lqr, name))
                for name in self._SNAPSHOT_ARRAYS
            },
            "agent_steps": self.agent_steps,
            "n_action_clips": self.n_action_clips,
            "frames": [f.copy() for f in self._frames],
        }

    def restore(self, snap: dict) -> None:
        """Restore the env from a snapshot."""
        self.lqr.state_rng.bit_generator.state = snap["state_rng"]
        self.lqr.noise_rng.bit_generator.state = snap["noise_rng"]
        self.lqr.env_steps = snap["env_steps"]
        self.lqr.episodes = snap["episodes"]
        self.lqr.initialized = snap["initialized"]
        for name in ("x", "x_0"):
            if snap[name] is not None:
                setattr(self.lqr, name, snap[name])
        self.lqr._y_cache = snap["_y_cache"]
        self.agent_steps = snap["agent_steps"]
        self.n_action_clips = snap["n_action_clips"]
        self._frames = deque(snap["frames"], maxlen=self.config.frame_stack)

    def render(self):
        """Show the scene."""
        if self.render_mode is None:
            return None

        rgb = self.renderer.rgb(
            float(self.lqr.x[0]), velocity=float(self.lqr.x[1]),
            thrust=self._last_thrust, angles=self.config.thruster_angles,
        )
        if self.render_mode == "rgb_array":
            return rgb

        import pygame

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((rgb.shape[1], rgb.shape[0]))
            pygame.display.set_caption("Pixel LQR Environment")
            self.clock = pygame.time.Clock()

        self.screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
        pygame.event.pump()
        self.clock.tick(max(1, round(1.0 / self.dt_agent)))
        pygame.display.flip()

    def close(self):
        """Close the pygame window, if one was opened."""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None


def build_pixel_lqr(seed_seq: np.random.SeedSequence) -> LQRInflated:
    """Create the LQR system for the pixel environment."""
    U_1, U_2 = cfg.thruster_lifts()
    Q, R = cfg.default_costs()

    return LQRInflated(LQRInflatedSettings(
        model=cfg.double_integrator_model(),
        Q=Q,
        R=R,
        max_initial_state=cfg.DEFAULT_MAX_INITIAL_STATE.copy(),
        resample_initial_state=True,
        noise_settings=None,
        seed_seq=seed_seq,
        dt=cfg.DEFAULT_DT,
        U_1=U_1,
        U_2=U_2,
    ))


def build_pixel_env(
    seed_seq: np.random.SeedSequence,
    horizon: int,
    render_mode: str | None = None,
) -> PixelLQREnv:
    """Build a chapter-6 environment from the constants in :mod:`config`.

    Raises
    ------
    ValueError
        If the frame is too coarse, or the action repeat too short.
    """
    lqr = build_pixel_lqr(seed_seq)
    n_thrusters = lqr.action_dim

    scene = SceneSpec(
        width=cfg.DEFAULT_FRAME_SIZE,
        height=cfg.DEFAULT_FRAME_SIZE,
        span=cfg.DEFAULT_VIEW_SCALE * float(lqr.max_initial_state[0]),
    )
    config = PixelEnvConfig(
        horizon=horizon,
        action_repeat=cfg.DEFAULT_ACTION_REPEAT,
        frame_stack=cfg.DEFAULT_FRAME_STACK,
        reward_scale=lqr.dt,
        scene=scene,
        thruster_angles=np.pi * np.arange(n_thrusters) / n_thrusters + cfg.DEFAULT_THRUSTER_PHASE,
    )

    # Slowest speed that still changes the frame between two agent steps.
    velocity_floor = scene.metres_per_pixel / (lqr.dt * config.action_repeat)
    if velocity_floor >= float(lqr.max_initial_state[1]):
        raise ValueError(
            f"Velocity floor {velocity_floor:.3f} m/s is not below the initial-velocity "
            f"bound; raise DEFAULT_FRAME_SIZE or DEFAULT_ACTION_REPEAT."
        )

    return PixelLQREnv(lqr, config, render_mode=render_mode)
