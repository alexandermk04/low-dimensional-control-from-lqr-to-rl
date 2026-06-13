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

from inflated_lqr import LQRInflated

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

    # x" = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    # Position, velocity
    x_0 = np.array([10, 0])

    INFLATED_INPUT_DIM = 20
    INFLATED_OUTPUT_DIM = 15

    # Orthogonal matrices to inflate the input and output spaces (Orthonormal lifts)
    U_1, _ = np.linalg.qr(np.random.randn(INFLATED_INPUT_DIM, A.shape[0]))
    U_2_full, _ = np.linalg.qr(np.random.randn(INFLATED_OUTPUT_DIM, B.shape[1]))
    U_2 = U_2_full.T

    model = StateSpaceModel(A, B, x_0)

    # Write identity matrices out, to easily modify them

    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    PARAMETRIC_NOISE_STD = 0.001
    CONTROL_NOISE_STD = 0.75
    MEASUREMENT_NOISE_STD = 0.01
    DISTURBANCE_NOISE_STD = 0.01
    DISTURBANCE_NOISE_TYPE = "gaussian"  

    # Noise
    parametric_noise = ParametricNoiseSettings(noise_std=PARAMETRIC_NOISE_STD, time_invariant=False)
    control_noise = ControlNoiseSettings(noise_std=CONTROL_NOISE_STD)
    measurement_noise = MeasurementNoiseSettings(noise_std=MEASUREMENT_NOISE_STD)
    disturbance_noise = DisturbanceNoiseSettings(noise_std=DISTURBANCE_NOISE_STD, disturbance_type=DISTURBANCE_NOISE_TYPE)

    param_noise_settings = LQRNoiseSettings(parametric_noise=parametric_noise)
    control_noise_settings = LQRNoiseSettings(control_noise=control_noise)
    measurement_noise_settings = LQRNoiseSettings(measurement_noise=measurement_noise)
    disturbance_noise_settings = LQRNoiseSettings(disturbance_noise=disturbance_noise)

    lqr = LQRInflated(model, Q, R, U_1, U_2, dt=0.01)
    lqr_param = LQRInflated(model, Q, R, U_1, U_2, noise_settings=param_noise_settings, dt=0.01)
    lqr_control = LQRInflated(model, Q, R, U_1, U_2, noise_settings=control_noise_settings, dt=0.01)
    lqr_measurement = LQRInflated(model, Q, R, U_1, U_2, noise_settings=measurement_noise_settings, dt=0.01)
    lqr_disturbance = LQRInflated(model, Q, R, U_1, U_2, noise_settings=disturbance_noise_settings, dt=0.01)

    true_lqr = LQRDiscrete(model, Q, R, dt=0.01)
    K_target, _ = baseline_solution(true_lqr)

    K_0 = np.linalg.pinv(U_2) @ K_target @ np.linalg.pinv(U_1)

    LIMIT = 500
    GRADIENT_LR = 1e-5
    TRAJECTORIES = 100
    ROLLOUT_LENGTH = 200 
    SMOOTHING_PARAM = 0.1

    mf_nat_settings = ModelFreePolicyGradientSettings(
        K_0=K_0, 
        m=TRAJECTORIES,  # Number of trajectoriesI hav
        l=ROLLOUT_LENGTH,  # Length of steps in each trajectory
        r=SMOOTHING_PARAM,  # Smoothing parameter
        gradient_steps=LIMIT,
        gradient_lr=GRADIENT_LR
    )

    mf_structured_settings = MFStructuredPolicySettings(
        K_0=K_0,
        beta=0.5,
        lambda_=5,
        stepsize=GRADIENT_LR,
        epsilon=1e-3,
        limit=LIMIT,
        m=TRAJECTORIES,
        l=ROLLOUT_LENGTH,
        r=SMOOTHING_PARAM
    )

    
    # parametric_ks, parametric_costs = model_free_natural_policy_gradient(lqr_param, mf_nat_settings)
    # print("Parametric Calculation Done")
    # control_ks, control_costs = model_free_natural_policy_gradient(lqr_control, mf_nat_settings)
    # print("Control Calculation Done")
    # measurement_ks, measurement_costs = model_free_natural_policy_gradient(lqr_measurement, mf_nat_settings)
    # print("Measurement Calculation Done")
    # disturbance_ks, disturbance_costs = model_free_natural_policy_gradient(lqr_disturbance, mf_nat_settings)
    # print("Disturbance Calculation Done")

    # parametric_ks, parametric_costs = structured_policy_iteration_mf(lqr=lqr_param, settings=mf_structured_settings)
    # print("Parametric Calculation Done")
    # control_ks, control_costs = structured_policy_iteration_mf(lqr=lqr_control, settings=mf_structured_settings)
    # print("Control Calculation Done")
    # measurement_ks, measurement_costs = structured_policy_iteration_mf(lqr=lqr_measurement, settings=mf_structured_settings)
    # print("Measurement Calculation Done")
    # disturbance_ks, disturbance_costs = structured_policy_iteration_mf(lqr=lqr_disturbance, settings=mf_structured_settings)
    # print("Disturbance Calculation Done")

    STEPS = 2500

    metadata = {
        "Inflated Input Dim": INFLATED_INPUT_DIM,
        "Inflated Output Dim": INFLATED_OUTPUT_DIM,
        "Gradient Steps": LIMIT,
        "Gradient LR": GRADIENT_LR,
        "Trajectories per Gradient Step": TRAJECTORIES,
        "Rollout Length": ROLLOUT_LENGTH,
        "Smoothing Parameter": SMOOTHING_PARAM,
        "Simulation Steps": STEPS
    }

    param_metadata = {
        "Parametric Noise Std": PARAMETRIC_NOISE_STD
    }
    control_metadata = {
        "Control Noise Std": CONTROL_NOISE_STD
    }
    measurement_metadata = {
        "Measurement Noise Std": MEASUREMENT_NOISE_STD
    }
    disturbance_metadata = {
        "Disturbance Noise Std": DISTURBANCE_NOISE_STD,
        "Disturbance Noise Type": DISTURBANCE_NOISE_TYPE
    }

    full_pure_k, full_pure_cost = model_free_natural_policy_gradient(lqr, mf_nat_settings)
    print("Full-rank pure Calculation Done")
    lqr.reset()
    low_pure_k, low_pure_cost = structured_policy_iteration_mf(lqr=lqr, settings=mf_structured_settings)
    print("Low-rank pure Calculation Done")

    simulation_dict = {
        "Full-rank Pure": simulate(lqr=lqr, K=full_pure_k[-1], n_steps=STEPS),
        "Low-rank Pure": simulate(lqr=lqr, K=low_pure_k[-1], n_steps=STEPS),
    }

    visualize_multiple_results(simulation_dict, metadata=metadata, dt=lqr.dt)
