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

from model_free_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_natural_policy_gradient
)

from structured_policy_iteration import (
    MFStructuredPolicySettings,
    structured_policy_iteration_mf
)


if __name__ == "__main__":
    # Double Integrator example implementation

    # x' = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    # Position, velocity
    x_0 = np.array([10, 0])

    model = StateSpaceModel(A, B, x_0)

    # Write identity matrices out, to easily modify them

    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    # Noise
    parametric_noise = ParametricNoiseSettings(noise_std=0.001, time_invariant=False)
    control_noise = ControlNoiseSettings(noise_std=0.1)
    measurement_noise = MeasurementNoiseSettings(noise_std=0.01)
    disturbance_noise = DisturbanceNoiseSettings(noise_std=0.01, disturbance_type='gaussian')

    param_noise_settings = LQRNoiseSettings(parametric_noise=parametric_noise)
    control_noise_settings = LQRNoiseSettings(control_noise=control_noise)
    measurement_noise_settings = LQRNoiseSettings(measurement_noise=measurement_noise)
    disturbance_noise_settings = LQRNoiseSettings(disturbance_noise=disturbance_noise)

    lqr = LQRDiscrete(model, Q, R, dt=0.01)
    lqr_param = LQRDiscrete(model, Q, R, noise_settings=param_noise_settings, dt=0.01)
    lqr_control = LQRDiscrete(model, Q, R, noise_settings=control_noise_settings, dt=0.01)
    lqr_measurement = LQRDiscrete(model, Q, R, noise_settings=measurement_noise_settings, dt=0.01)
    lqr_disturbance = LQRDiscrete(model, Q, R, noise_settings=disturbance_noise_settings, dt=0.01)

    K_0 = np.array([[0.1, 0.5]])

    GRADIENT_STEPS = 200
    TRAJECTORIES = 100
    ROLLOUT_LENGTH = 200 
    SMOOTHING_PARAM = 0.01

    mf_nat_settings = ModelFreePolicyGradientSettings(
        K_0=K_0, 
        m=TRAJECTORIES,  # Number of trajectories
        l=ROLLOUT_LENGTH,  # Length of steps in each trajectory
        r=SMOOTHING_PARAM,  # Smoothing parameter
        gradient_steps=GRADIENT_STEPS,
        gradient_lr=0.001 
    )

    mf_structured_settings = MFStructuredPolicySettings(
        K_0=K_0,
        beta=0.5,
        lambda_=0.1,
        stepsize=1e-4,
        epsilon=1e-3, 
        limit=1000,
        m=TRAJECTORIES,
        l=ROLLOUT_LENGTH,
        r=SMOOTHING_PARAM
    )

    # Model Free
    # pure_k, pure_cost = model_free_natural_policy_gradient(lqr, mf_nat_settings)
    # print("Pure Calculation Done")
    # parametric_ks, parametric_costs = model_free_natural_policy_gradient(lqr_param, mf_nat_settings)
    # print("Parametric Calculation Done")
    # control_ks, control_costs = model_free_natural_policy_gradient(lqr_control, mf_nat_settings)
    # print("Control Calculation Done")
    # measurement_ks, measurement_costs = model_free_natural_policy_gradient(lqr_measurement, mf_nat_settings)
    # print("Measurement Calculation Done")
    # disturbance_ks, disturbance_costs = model_free_natural_policy_gradient(lqr_disturbance, mf_nat_settings)
    # print("Disturbance Calculation Done")

    # Structured Policy Iteration
    pure_k, pure_cost = structured_policy_iteration_mf(lqr=lqr, settings=mf_structured_settings)
    print("Pure Calculation Done")
    parametric_ks, parametric_costs = structured_policy_iteration_mf(lqr=lqr_param, settings=mf_structured_settings)
    print("Parametric Calculation Done")
    control_ks, control_costs = structured_policy_iteration_mf(lqr=lqr_control, settings=mf_structured_settings)
    print("Control Calculation Done")
    measurement_ks, measurement_costs = structured_policy_iteration_mf(lqr=lqr_measurement, settings=mf_structured_settings)
    print("Measurement Calculation Done")
    disturbance_ks, disturbance_costs = structured_policy_iteration_mf(lqr=lqr_disturbance, settings=mf_structured_settings)
    print("Disturbance Calculation Done")

    STEPS = 2500

    simulation_dict = {
        'Pure': simulate(lqr=lqr, K=pure_k[-1], n_steps=STEPS),
        'Parametric': simulate(lqr=lqr_param, K=parametric_ks[-1], n_steps=STEPS),
        'Control': simulate(lqr=lqr_control, K=control_ks[-1], n_steps=STEPS),
        'Measurement': simulate(lqr=lqr_measurement, K=measurement_ks[-1], n_steps=STEPS),
        'Disturbance': simulate(lqr=lqr_disturbance, K=disturbance_ks[-1], n_steps=STEPS)
    }

    visualize_multiple_results(simulation_dict, dt=lqr.dt)
