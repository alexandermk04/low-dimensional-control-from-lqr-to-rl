"""The test protocol and trajectories we later draw figures from.

The singular protocol: ``settings.TEST_EPISODES`` rollouts of
``settings.TEST_HORIZON`` steps. Every episode of every seed is pooled. 
Noise is only added, if the experiment requires it.
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
    "evaluate_policy",
    "figure_trajectory",
    "runaway_episodes",
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


def runaway_episodes(problem: Problem, final_states: np.ndarray) -> int:
    """How many episodes ended with the mass outside the initial position range.
    Treated as unstable.

    Parameters
    ----------
    problem : Problem
        Only the initial-state box is used.
    final_states : np.ndarray
        True states the episodes ended in, of shape ``(episodes, n)``.

    Returns
    -------
    int
    """
    position = final_states[:, 0]
    outside = ~(np.abs(position) <= problem.max_initial_state[0])
    return int(np.sum(outside))


def evaluate_policy(
    problem: Problem,
    policy: Policy,
    noise_settings: LQRNoiseSettings | None = None,
    episodes: int | None = None,
    horizon: int | None = None,
) -> TestResult:
    """Score a policy under the test protocol.

    Parameters
    ----------
    problem : Problem
        The experiment setting the policy was trained on.
    policy : Policy
        The policy mapping an observation to a control output.
    noise_settings : LQRNoiseSettings or None, optional
        Whether to apply noise to the plan
    episodes, horizon : int or None, optional
        Override the protocol, for possible shorter pre-tests.

    Returns
    -------
    TestResult
        Its ``summary`` carries ``test_cost_mean``, ``test_episodes``,
        ``test_horizon`` and ``runaway_episodes``; its ``returns`` contains
        the per-episode costs. ``runaway_episodes`` is ``None`` for a linear
        gain, whose stability the spectral radius already decides.
    """
    episodes = settings.TEST_EPISODES if episodes is None else episodes
    horizon = settings.TEST_HORIZON if horizon is None else horizon

    scorer = build_plant(problem, "test", noise_settings=noise_settings)
    scorer.reset()

    returns, finals = monte_carlo_returns(
        scorer, policy, n_episodes=episodes, horizon=horizon, return_final=True
    )

    return TestResult(
        summary={
            "test_cost_mean": float(np.mean(returns)),
            "test_episodes": episodes,
            "test_horizon": horizon,
            "runaway_episodes": (
                None if isinstance(policy, LinearPolicy)
                else runaway_episodes(problem, finals)
            ),
        },
        returns=returns,
    )


def figure_trajectory(
    problem: Problem,
    policy: Policy,
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
        "figure",
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
