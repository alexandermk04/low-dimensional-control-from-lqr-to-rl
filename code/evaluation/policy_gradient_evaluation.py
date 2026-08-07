import numpy as np

from visualization import visualize_multiple_results

from lqr_base.simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    LQRDiscreteSettings,
    simulate,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from lqr_solvers.mb_policy_gradient import (
    ModelBasedPolicyGradientSettings,
    gradient_descent_discrete as model_based_gradient
)

from lqr_solvers.mf_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_policy_gradient, 
    model_free_natural_policy_gradient
)

from visualization import visualize_costs, visualize_Ks

def compare_all():

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])

    model = StateSpaceModel(A, B)

    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])

    settings = LQRDiscreteSettings(
        model=model,
        Q=Q,
        R=R,
        max_initial_state=np.array([10, 1]),
        resample_initial_state=False,
        noise_settings=None,
        dt=0.01
    )

    lqr = LQRDiscrete(settings=settings)

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
    print("Baseline done.")
    mb_ks, mb_costs = model_based_gradient(lqr=lqr, settings=mb_settings)
    print("Model-based done.")
    mf_ks, mf_costs = model_free_policy_gradient(lqr=lqr, settings=mf_settings)
    print("Model-free done.")
    mf_nat_ks, mf_nat_costs = model_free_natural_policy_gradient(lqr=lqr, settings=mf_nat_settings)
    print("Natural model-free done.")

    costs_dict = {
        'Model-Based': mb_costs,
        'Model-Free': mf_costs,
        'Model-Free Natural': mf_nat_costs
    }

    STEPS = 2500

    simulation_dict = {
        'Baseline': simulate(lqr=lqr, K=baseline_k, n_steps=STEPS),
        'Model-Based': simulate(lqr=lqr, K=mb_ks[-1], n_steps=STEPS),
        'Model-Free': simulate(lqr=lqr, K=mf_ks[-1], n_steps=STEPS),
        'Model-Free Natural': simulate(lqr=lqr, K=mf_nat_ks[-1], n_steps=STEPS)
    }

    visualize_multiple_results(results=simulation_dict,
                               cost_trajectories=costs_dict,
                               baseline_cost=baseline_cost,
                                dt=lqr.dt)

if __name__ == "__main__":
    compare_all()