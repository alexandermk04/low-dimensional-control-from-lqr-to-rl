"""Plotting helpers shared by the scripts in :mod:`evaluation`.

These show plots directly based on the input data.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from basetypes import SimulationResult

# Path for saving the figures
FIGURE_ROOT = "results/figures"


def _figure_path(name: str) -> str:
    """Absolute path for a figure, creating the directory if it doesn't exist."""
    os.makedirs(FIGURE_ROOT, exist_ok=True)
    return os.path.join(FIGURE_ROOT, f"{name}.png")


def relative_cost_error(cost_estimate, baseline_cost):
    """Suboptimality of a cost relative to the analytic optimum.

    Parameters
    ----------
    cost_estimate : float
        Cost achieved by the policy under evaluation.
    baseline_cost : float
        Optimal cost from
        :func:`~lqr_base.simple_lqr.get_optimal_K_and_cost_discrete`.

    Returns
    -------
    float
        ``(cost_estimate - baseline_cost) / baseline_cost``. 
    """
    return (cost_estimate - baseline_cost) / baseline_cost

def feedback_control_distance(K, K_baseline):
    """Frobenius distance between a gain and the optimal one.

    Parameters
    ----------
    K : np.ndarray
        Gain under evaluation. Must live in the same space as ``K_baseline``.
    K_baseline : np.ndarray
        Optimal gain of the same shape.

    Returns
    -------
    float
        ``||K - K_baseline||_F``.
    """
    return np.linalg.norm(K - K_baseline, ord='fro')

def visualize_costs(baseline_cost, costs_dict: dict[str, list[float]]):
    """Plot absolute and relative cost against the iteration count.

    Two panels side by side: the raw cost curves with the optimum drawn in, and
    the same curves as relative error.

    Parameters
    ----------
    baseline_cost : float
        Optimal cost, drawn as a dashed horizontal reference.
    costs_dict : dict[str, list[float]]
        Legend label to cost-per-iteration series.

    Returns
    -------
    None
        Shows the figure.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for label, costs in costs_dict.items():
        plt.semilogy(costs, label=label)
    plt.axhline(y=baseline_cost, color='r', linestyle='--', label='Baseline Cost')
    plt.xlabel('Gradient Step')
    plt.ylabel('Cost')
    plt.title('Estimated Costs At Different Gradient Steps')
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, costs in costs_dict.items():
        relative_errors = [relative_cost_error(cost, baseline_cost) for cost in costs]
        plt.semilogy(relative_errors, label=label)
    plt.xlabel('Gradient Step')
    plt.ylabel('Relative Cost Error')
    plt.title('Relative Cost Error At Different Gradient Steps')
    plt.legend()
    plt.show()

def visualize_Ks(K_baseline, k_dict: dict[str, list[np.ndarray]]):
    """Plot how far each solver's gain sits from the optimum over time.

    Parameters
    ----------
    K_baseline : np.ndarray
        Optimal gain.
    k_dict : dict[str, list[np.ndarray]]
        Legend label to the sequence of gains produced by that solver. Each
        gain must be comparable to ``K_baseline``.

    Returns
    -------
    None
        Shows the figure.
    """
    plt.figure(figsize=(10, 5))

    for label, Ks in k_dict.items():
        distances = [feedback_control_distance(K, K_baseline) for K in Ks]
        plt.plot(distances, label=label)

    plt.xlabel('Gradient Step')
    plt.ylabel('Distance to Baseline K')
    plt.title('Distance of Estimated K to Baseline K at Different Gradient Steps')
    plt.legend()
    plt.show()


def visualize_result(result: SimulationResult, dt=1):
    """Plot a single double-integrator trajectory.

    Parameters
    ----------
    result : SimulationResult
        Trajectory from :func:`~lqr_base.simple_lqr.simulate`.
    dt : float, optional
        Sampling interval, used to label the x-axis in seconds.

    Returns
    -------
    None
        Shows the figure.
    """
    time_sec = np.arange(result.x_traj.shape[0]) * dt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_sec, result.x_traj[:, 0], label='Position')
    plt.plot(time_sec, result.x_traj[:, 1], label='Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time_sec, result.u_traj[:, 0], label='Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Control')
    plt.legend()

    plt.show()

def visualize_multiple_results(
    results: dict[str, SimulationResult],
    cost_trajectories: dict[str, list[float]],
    baseline_cost: float,
    ranks_dict: dict[str, int] = None,
    metadata: dict[str, str] = None,
    save_to: str = None,
    dt=1,
):
    """Six-panel overview comparing several solvers on one problem.

    Parameters
    ----------
    results : dict[str, SimulationResult]
        Legend label to trajectory, one per policy being compared.
    cost_trajectories : dict[str, list[float]]
        Legend label to the true cost after each solver iteration. 
    baseline_cost : float
        Optimal cost, drawn as a dashed reference line.
    ranks_dict : dict[str, int], optional
        Legend label to a rank-per-iteration series.
    metadata : dict[str, str], optional
        Key/value pairs rendered into the sixth panel.
    save_to : str, optional
        Filename stem. When given, the figure is written to
        ``results/figures/<save_to>.png`` instead of being shown.
    dt : float, optional
        Sampling interval, for labelling the time axes in seconds.

    Returns
    -------
    None
        Shows or saves the figure.
    """

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.title('Position')
    for label, result in results.items():
        time_sec = np.arange(result.x_traj.shape[0]) * dt
        plt.plot(time_sec, result.x_traj[:, 0], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.title('Velocity')
    for label, result in results.items():
        time_sec = np.arange(result.x_traj.shape[0]) * dt
        plt.plot(time_sec, result.x_traj[:, 1], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.title('Control')
    for label, result in results.items():
        time_sec = np.arange(result.u_traj.shape[0]) * dt
        plt.plot(time_sec, result.u_traj[:, 0], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Control')
    plt.legend()

    # True cost over gradient iterations
    plt.subplot(2, 3, 4)
    plt.title('True Cost')
    plt.axhline(y=baseline_cost, linestyle='--', label='Optimal Cost')
    for label, costs in cost_trajectories.items():
        plt.semilogy(costs, label=f'{label}')
    plt.xlabel('Gradient Step')
    plt.ylabel('Cost')
    plt.legend()

    if ranks_dict is not None:
        plt.subplot(2, 3, 5)
        plt.title('Rank of K over SPI Iterations')
        for label, ranks in ranks_dict.items():
            plt.plot(ranks, label=f'{label}')
        plt.xlabel('Iteration')
        plt.ylabel('Rank')
        plt.legend()

    # For adding run metadata
    plt.subplot(2, 3, 6)
    plt.axis('off')

    if metadata:
        text_content = ""
        for key, value in metadata.items():
                text_content += f"{key}: {value}\n"
        plt.text(0.5, 0.5, text_content,
                     fontsize=12,
                     horizontalalignment='center',
                     verticalalignment='center'
                     )

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(_figure_path(save_to))
    else:
        plt.show()


def visualize_spi(
    k_ranks: dict[str, list[int]],
    cost_trajectories: dict[str, list[float]],
    baseline_cost: float,
    simulation_dict: dict[str, SimulationResult],
    seed: int,
    dt=1,
    save_to: str = None,
):
    """Four-panel figure focused on structured policy iteration.

    Parameters
    ----------
    k_ranks : dict[str, list[int]]
        Legend label to the numerical rank of the gain after each iteration.
    cost_trajectories : dict[str, list[float]]
        Legend label to the true cost after each iteration.
    baseline_cost : float
        Optimal cost, drawn as a dashed reference line.
    simulation_dict : dict[str, SimulationResult]
        Legend label to trajectory, shown as position and velocity.
    dt : float, optional
        Sampling interval.
    save_to : str, optional
        Filename stem; see :func:`visualize_multiple_results`.

    Returns
    -------
    None
        Shows or saves the figure.
    """
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'SPI Results (Seed: {seed})', fontsize=16)

    plt.subplot(2, 2, 1)
    plt.title('Rank of K over SPI Iterations')
    for label, ranks in k_ranks.items():
        plt.plot(ranks, label=f'{label}')
    plt.xlabel('SPI Iteration')
    plt.ylabel('Rank')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('True Cost over SPI Iterations')
    plt.axhline(y=baseline_cost, linestyle='--', label='Optimal Cost')
    for label, costs in cost_trajectories.items():
        plt.semilogy(costs, label=f'{label}')
    plt.xlabel('SPI Iteration')
    plt.ylabel('Cost')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Position Trajectories')
    for label, result in simulation_dict.items():
        time_sec = np.arange(result.x_traj.shape[0]) * dt
        plt.plot(time_sec, result.x_traj[:, 0], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title('Velocity Trajectories')
    for label, result in simulation_dict.items():
        time_sec = np.arange(result.x_traj.shape[0]) * dt
        plt.plot(time_sec, result.x_traj[:, 1], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(_figure_path(save_to))
    else:
        plt.show()
