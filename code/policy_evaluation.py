import numpy as np
import matplotlib.pyplot as plt

from simple_lqr import StateSpaceModel, LQRDiscrete, simulate, visualize

from simple_lqr import (
    StateSpaceModel,
    SimulationResult,
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

def feedback_control_distance(K_1, K_2):
    return np.linalg.norm(K_1 - K_2, ord='fro')