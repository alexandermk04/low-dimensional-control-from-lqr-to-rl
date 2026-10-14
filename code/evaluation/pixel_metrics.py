"""Scoring for policies that act in the pixel environment.
"""

import numpy as np

from basetypes import SimulationResult

__all__ = [
    "StatePolicy",
    "SB3EnvPolicy",
    "env_monte_carlo_returns",
    "env_monte_carlo_cost",
    "rollout_env",
]


class StatePolicy:
    """Wraps a state-feedback policy so it reads ``info["state"]``."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, observation: np.ndarray, info: dict) -> np.ndarray:
        return self.policy(info["state"])


class SB3EnvPolicy:
    """Wraps a trained Stable-Baselines3 model."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, observation: np.ndarray, info: dict) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action


def env_monte_carlo_returns(env, policy, n_episodes: int, horizon: int | None = None
                            ) -> tuple[np.ndarray, dict]:
    """Undiscounted cost of each individual episode.

    Parameters
    ----------
    env : PixelLQREnv
        Environment to roll out; restored afterwards.
    policy : callable
        ``policy(observation, info) -> action``.
    n_episodes : int
        Rollouts to average over.
    horizon : int or None, optional
        Agent steps per rollout. Defaults to the environment's horizon.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(returns, diagnostics)``: the cost per episode, of shape ``(n_episodes,)``,
        and ``thruster_energy_mean``, ``null_waste_mean``, ``saturation_rate``,
        ``off_screen_rate`` averaged over steps and ``terminated_rate`` over episodes.
    """
    horizon = env.config.horizon if horizon is None else horizon
    snap = env.snapshot()

    returns = np.empty(n_episodes)
    energy, waste, saturated, off_screen, steps, terminated_episodes = 0.0, 0.0, 0, 0, 0, 0

    try:
        for episode in range(n_episodes):
            observation, info = env.reset()
            total = 0.0

            for _ in range(horizon):
                observation, _, terminated, _, info = env.step(policy(observation, info))

                total += info["stage_cost"]
                energy += info["thruster_energy"]
                waste += info["null_waste"]
                saturated += int(info["clipped"])
                off_screen += int(info["off_screen"])
                steps += 1

                if terminated:
                    terminated_episodes += 1
                    break

            returns[episode] = total
    finally:
        env.restore(snap)

    steps = max(steps, 1)

    return returns, {
        "thruster_energy_mean": energy / steps,
        "null_waste_mean": waste / steps,
        "saturation_rate": saturated / steps,
        "off_screen_rate": off_screen / steps,
        "terminated_rate": terminated_episodes / n_episodes,
    }


def env_monte_carlo_cost(env, policy, n_episodes: int, horizon: int | None = None
                         ) -> tuple[float, float]:
    """Expected undiscounted episode cost.

    Returns
    -------
    tuple[float, float]
        ``(mean, standard_error)`` of :func:`env_monte_carlo_returns`.
    """
    returns, _ = env_monte_carlo_returns(env, policy, n_episodes, horizon)
    stderr = float(np.std(returns, ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0

    return float(np.mean(returns)), stderr


def rollout_env(env, policy, n_agent_steps: int) -> SimulationResult:
    """Simulate interactions in the environment; env restored afterwards."""
    snap = env.snapshot()
    states, controls = [], []

    try:
        observation, info = env.reset()
        states.append(info["state"])

        for _ in range(n_agent_steps):
            observation, _, terminated, truncated, info = env.step(policy(observation, info))
            states.append(info["state"])
            controls.append([info["u"]])
            if terminated or truncated:
                break
    finally:
        env.restore(snap)

    return SimulationResult(
        x_traj=np.asarray(states[:-1], dtype=float),
        u_traj=np.asarray(controls, dtype=float),
    )
