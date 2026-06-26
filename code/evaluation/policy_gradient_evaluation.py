import numpy as np
import matplotlib.pyplot as plt

from visualization import visualize_multiple_results

from noise import (
    ParametricNoiseSettings,
    ControlNoiseSettings,
    MeasurementNoiseSettings,
    DisturbanceNoiseSettings,
    LQRNoiseSettings
)

from simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    simulate,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from lqr_policy_gradient import (
    ModelBasedPolicyGradientSettings,
    gradient_descent_discrete as model_based_gradient
)

from model_free_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_policy_gradient, 
    model_free_natural_policy_gradient
)

def relative_cost_error(cost_estimate, baseline_cost):
    return (cost_estimate - baseline_cost) / baseline_cost

def feedback_control_distance(K, K_baseline):
    return np.linalg.norm(K - K_baseline, ord='fro')

def visualize_costs(baseline_cost, costs_dict: dict[str, list[float]]):
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
    plt.figure(figsize=(10, 5))

    for label, Ks in k_dict.items():
        distances = [feedback_control_distance(K, K_baseline) for K in Ks]
        plt.plot(distances, label=label)
    
    plt.xlabel('Gradient Step')
    plt.ylabel('Distance to Baseline K')
    plt.title('Distance of Estimated K to Baseline K at Different Gradient Steps')
    plt.legend()
    plt.show()

def compare_all():

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    x_0 = np.array([10, 0])

    model = StateSpaceModel(A, B, x_0)

    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])

    lqr = LQRDiscrete(model, Q, R)

    K_0 = np.array([[0.1, 0.5]])

    GRADIENT_STEPS = 200
    ROLLOUT_LENGTH = 200 # SHOW: 100 underestimates the cost

    mb_settings = ModelBasedPolicyGradientSettings(
        K_0=K_0,
        gradient_steps=GRADIENT_STEPS,
        gradient_lr=0.001
    )

    mf_settings = ModelFreePolicyGradientSettings(
        K_0=K_0, 
        m=40,  # Number of trajectories
        l=ROLLOUT_LENGTH,  # Length of steps in each trajectory
        r=0.1,  # Smoothing parameter
        gradient_steps=GRADIENT_STEPS,
        gradient_lr=1e-4
    )

    mf_nat_settings = ModelFreePolicyGradientSettings(
        K_0=K_0, 
        m=40,  # Number of trajectories
        l=ROLLOUT_LENGTH,  # Length of steps in each trajectory
        r=0.1,  # Smoothing parameter
        gradient_steps=GRADIENT_STEPS,
        gradient_lr=0.001 # SHOW: 0.01 becomes unstable
    )

    baseline_k, baseline_cost = baseline_solution(lqr=lqr)
    mb_ks, mb_costs = model_based_gradient(lqr=lqr, settings=mb_settings)
    mf_ks, mf_costs = model_free_policy_gradient(lqr=lqr, settings=mf_settings)
    mf_nat_ks, mf_nat_costs = model_free_natural_policy_gradient(lqr=lqr, settings=mf_nat_settings)

    costs_dict = {
        'Model-Based': mb_costs,
        'Model-Free': mf_costs,
        'Model-Free Natural': mf_nat_costs
    }

    k_dict = {
        'Model-Based': mb_ks,
        'Model-Free': mf_ks,
        'Model-Free Natural': mf_nat_ks
    }

    visualize_costs(baseline_cost, costs_dict)
    visualize_Ks(baseline_k, k_dict)

    STEPS = 2500

    simulation_dict = {
        'Baseline': simulate(lqr=lqr, K=baseline_k, n_steps=STEPS),
        'Model-Based': simulate(lqr=lqr, K=mb_ks[-1], n_steps=STEPS),
        'Model-Free': simulate(lqr=lqr, K=mf_ks[-1], n_steps=STEPS),
        'Model-Free Natural': simulate(lqr=lqr, K=mf_nat_ks[-1], n_steps=STEPS)
    }

    visualize_multiple_results(simulation_dict, dt=lqr.dt)

if __name__ == "__main__":
    compare_all()