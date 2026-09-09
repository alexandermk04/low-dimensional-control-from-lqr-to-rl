"""The test protocol and trajectories we later draw figures from.

The singular protocol: ``settings.TEST_EPISODES`` rollouts of
``settings.TEST_HORIZON`` steps, accumulated cost averaged over episodes and
later over seeds. Noise is only attached here, if the experiment requires it.
"""

from dataclasses import dataclass

import numpy as np

import settings
from evaluation.metrics import LinearPolicy, Policy, monte_carlo_returns
from experiments.problem import Problem, build_plant
from seeding import sub_rng
from lqr_base.noise import LQRNoiseSettings
from lqr_base.simple_lqr import SimulationResult

__all__ = [
    "TestResult",
    "test_cost",
    "evaluate_policy",
    "figure_trajectory",
    "shared_initial_state",
]


@dataclass
class TestResult:
    """What one run of the test protocol produced.

    Attributes
    ----------
    summary : dict
        The scalar entries that go into a record's summary.
    returns : np.ndarray
        Accumulated cost of every individual episode, of shape
        ``(test_episodes,)``.
    """

    summary: dict
    returns: np.ndarray


def shared_initial_state(problem: Problem) -> np.ndarray:
    """A singular initial state such that simulation figures compare correctly.

    Parameters
    ----------
    problem : Problem
        Only state dimension and initial-state box are used.

    Returns
    -------
    np.ndarray
        Of shape ``(n,)``.
    """
    bound = problem.max_initial_state
    return sub_rng(settings.SHARED_SEED, "figure_x0").uniform(-bound, bound)


def evaluate_policy(
    problem: Problem,
    policy: Policy,
    purpose: str,
    noise_settings: LQRNoiseSettings | None = None,
    episodes: int | None = None,
    horizon: int | None = None,
) -> dict:
    """Score a policy under the test protocol.

    Parameters
    ----------
    problem : Problem
        The experiment setting the policy was trained on.
    policy : Policy
        The policy mapping an observation to a control output.
    purpose : str
        Seeds the test plant.
    noise_settings : LQRNoiseSettings or None, optional
        Whether to apply noise to the plan
    episodes, horizon : int or None, optional
        Override the protocol, for possible shorter pre-tests.

    Returns
    -------
    TestResult
        Its ``summary`` carries ``test_cost_mean``, ``test_cost_std``,
        ``test_cost_stderr``, ``test_episodes`` and ``test_horizon``; its
        ``returns`` contains the per-episode costs.
    """
    episodes = settings.TEST_EPISODES if episodes is None else episodes
    horizon = settings.TEST_HORIZON if horizon is None else horizon

    scorer = build_plant(problem, f"test:{purpose}", noise_settings=noise_settings)
    scorer.reset()

    returns = monte_carlo_returns(
        scorer, policy, n_episodes=episodes, horizon=horizon
    )

    std = float(np.std(returns, ddof=1)) if episodes > 1 else 0.0

    return TestResult(
        summary={
            "test_cost_mean": float(np.mean(returns)),
            "test_cost_std": std,
            "test_cost_stderr": std / np.sqrt(episodes) if episodes > 1 else 0.0,
            "test_episodes": episodes,
            "test_horizon": horizon,
        },
        returns=returns,
    )


def test_cost(problem: Problem, K: np.ndarray, purpose: str, **kwargs) -> TestResult:
    """:func:`evaluate_policy` for a gain matrix in the controller's space."""
    return evaluate_policy(problem, LinearPolicy(K), purpose, **kwargs)


def figure_trajectory(
    problem: Problem,
    policy: Policy,
    purpose: str,
    noise_settings: LQRNoiseSettings | None = None,
    n_steps: int | None = None,
) -> SimulationResult:
    """Roll one trajectory out from the shared initial state.

    Parameters
    ----------
    problem : Problem
        The experiment setting the policy was trained on.
    policy : Policy
        The controller to roll out.
    purpose : str
        Seeds the plant's noise stream.
    noise_settings : LQRNoiseSettings or None, optional
        Noise settings to the plant
    n_steps : int or None, optional
        Steps to display, defaults to ``settings.FIGURE_HORIZON``.

    Returns
    -------
    SimulationResult
        ``x_traj`` of shape ``(n_steps, n)`` and ``u_traj`` of shape
        ``(n_steps, m)``.
    """
    n_steps = settings.FIGURE_HORIZON if n_steps is None else n_steps

    plant = build_plant(
        problem,
        f"figure:{purpose}",
        noise_settings=noise_settings,
        resample_initial_state=False,
    )
    plant.x_0 = shared_initial_state(problem)
    plant.reset()

    x_traj = np.empty((n_steps, plant.A.shape[0]))
    u_traj = np.empty((n_steps, plant.B.shape[1]))

    for step in range(n_steps):
        result = plant.apply_control(policy(plant.observe()))
        x_traj[step] = result.x_t
        u_traj[step] = result.u_t

    return SimulationResult(x_traj=x_traj, u_traj=u_traj)
