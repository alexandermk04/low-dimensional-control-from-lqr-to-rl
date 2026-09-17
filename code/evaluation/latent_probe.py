"""Measuring how well a policy restores the true state.

A ridge regression from the latent to the true state, scored on held-out data.
Ridge regularization keeps wide latents from fitting noise, and the split is by
*episode*, because consecutive steps of one episode are nearly identical and a
random split would leak the test set into training.
"""

import numpy as np
import torch

__all__ = ["collect_latents", "linear_probe_r2"]


def collect_latents(env, extractor, policy, n_episodes: int, horizon: int | None = None):
    """Pair an extractor's latent with the true state, step by step.

    Parameters
    ----------
    env : PixelLQREnv
        Environment to roll out; restored afterwards.
    extractor : PixelExtractor
        Architecture whose ``latent`` is probed.
    policy : callable
        ``policy(observation, info) -> action``, used to visit states.
    n_episodes : int
        Episodes to collect.
    horizon : int or None, optional
        Steps per episode. Defaults to the environment's horizon.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Latents ``(T, d)``, true states ``(T, n)`` and episode indices ``(T,)``.
    """
    horizon = env.config.horizon if horizon is None else horizon
    snap = env.snapshot()
    latents, states, episodes = [], [], []

    try:
        for episode in range(n_episodes):
            observation, info = env.reset()

            for _ in range(horizon):
                observation, _, terminated, truncated, info = env.step(policy(observation, info))

                with torch.no_grad():
                    z = extractor.latent(torch.as_tensor(observation[None]))

                latents.append(z.cpu().numpy()[0])
                states.append(info["state"])
                episodes.append(episode)

                if terminated or truncated:
                    break
    finally:
        env.restore(snap)

    return np.asarray(latents, dtype=float), np.asarray(states, dtype=float), np.asarray(episodes)


def linear_probe_r2(Z: np.ndarray, S: np.ndarray, episodes: np.ndarray,
                    ridge: float = 1e-3, test_fraction: float = 0.3) -> dict:
    """Held-out R2 of a ridge regression from latents ``Z`` to states ``S``.

    The last ``test_fraction`` of episodes is held out. Scores are not clipped,
    so a latent that carries nothing can score below zero.

    Returns
    -------
    dict
        ``r2_position``, ``r2_velocity`` and ``latent_dim``.
    """
    unique = np.unique(episodes)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    is_test = np.isin(episodes, unique[-n_test:])

    if is_test.all() or not is_test.any():
        raise ValueError("The episode split left one side empty; collect more episodes.")

    Z_train, S_train = Z[~is_test], S[~is_test]

    mean = Z_train.mean(axis=0)
    scale = Z_train.std(axis=0)
    scale[scale < 1e-12] = 1.0

    def design(matrix):
        return np.hstack([(matrix - mean) / scale, np.ones((matrix.shape[0], 1))])

    A = design(Z_train)
    weights = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ S_train)

    S_test = S[is_test]
    residual = ((S_test - design(Z[is_test]) @ weights) ** 2).sum(axis=0)
    total = ((S_test - S_train.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-30)

    return {
        "r2_position": float(r2[0]),
        "r2_velocity": float(r2[1]),
        "latent_dim": int(Z.shape[1]),
    }

def latent_extra(model, env, policy) -> dict:
    """How well a policy encodes the true state."""
    Z, S, episodes = collect_latents(env, model.policy.features_extractor, policy, n_episodes=20)
    return linear_probe_r2(Z, S, episodes)
