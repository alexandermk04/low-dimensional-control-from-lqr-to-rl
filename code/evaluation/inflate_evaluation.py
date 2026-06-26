import numpy as np
import time

from visualization import visualize_multiple_results
from evaluation.policy_gradient_evaluation import visualize_costs

from lqr_base.noise import (
    ParametricNoiseSettings,
    ControlNoiseSettings,
    MeasurementNoiseSettings,
    DisturbanceNoiseSettings,
    LQRNoiseSettings
)

from lqr_base.simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    simulate,
    true_cost,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from lqr_base.inflated_lqr import LQRInflated

from basetypes import SimulationResult

from lqr_solvers.mf_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_natural_policy_gradient
)

from lqr_solvers.structured_policy_iteration import (
    MFStructuredPolicySettings,
    structured_policy_iteration_mf
)


class InflationExperiment:

    def __init__(
        self,
        model: StateSpaceModel,
        Q: np.ndarray,
        R: np.ndarray,
        U_1: np.ndarray,
        U_2: np.ndarray,
        K_0: np.ndarray,
        inflated_input_dim: int,
        inflated_output_dim: int,
        nat_settings: ModelFreePolicyGradientSettings,
        structured_settings: MFStructuredPolicySettings,
        dt: float = 0.01,
        sim_steps: int = 2500,
    ):
        self.model = model
        self.Q = Q
        self.R = R
        self.dt = dt
        self.sim_steps = sim_steps

        self.inflated_input_dim = inflated_input_dim
        self.inflated_output_dim = inflated_output_dim

        self.U_1 = U_1
        self.U_2 = U_2

        self.K_0 = K_0

        self.nat_settings = nat_settings
        self.structured_settings = structured_settings

        self.base_metadata = {
            "Inflated Input Dim": inflated_input_dim,
            "Inflated Output Dim": inflated_output_dim,
            "Gradient Steps": nat_settings.gradient_steps,
            "Gradient LR": nat_settings.gradient_lr,
            "Lambda SPI": structured_settings.lambda_,
            "Beta SPI": structured_settings.beta,
            "Trajectories per Gradient Step": nat_settings.m,
            "Rollout Length": nat_settings.l,
            "Smoothing Parameter": nat_settings.r,
            "Simulation Steps": sim_steps,
        }

    def run_comparison(
        self,
        label: str,
        noise_settings: LQRNoiseSettings | None = None,
        extra_metadata: dict | None = None,
        visualize: bool = True,
    ):
        
        base_lqr = LQRDiscrete(
            self.model, self.Q, self.R, noise_settings=noise_settings, dt=self.dt
        )
        lqr = LQRInflated(
            self.model, self.Q, self.R, self.U_1, self.U_2,
            noise_settings=noise_settings, dt=self.dt,
        )

        base_k, base_cost = baseline_solution(base_lqr)
        print(f"[{label}] Baseline solution done")
        full_ks, _ = model_free_natural_policy_gradient(lqr=lqr, settings=self.nat_settings)
        print(f"[{label}] Full-rank calculation done")
        low_ks, _ = structured_policy_iteration_mf(lqr=lqr, settings=self.structured_settings)
        print(f"[{label}] Low-rank calculation done")

        cost_trajectories = {
            'Full-rank': [true_cost(lqr, self.U_2 @ K @ self.U_1) for K in full_ks],
            'Low-rank': [true_cost(lqr, self.U_2 @ K @ self.U_1) for K in low_ks],
        }

        simulation_dict = {
            f"Baseline {label}": simulate(lqr=base_lqr, K=base_k, n_steps=self.sim_steps),
            f"Full-rank {label}": simulate(lqr=lqr, K=full_ks[-1], n_steps=self.sim_steps),
            f"Low-rank {label}": simulate(lqr=lqr, K=low_ks[-1], n_steps=self.sim_steps),
        }

        ranks_dict = {
            f"MatrixRank Full {label}": np.linalg.matrix_rank(full_ks),
            f"MatrixRank Low {label}": np.linalg.matrix_rank(low_ks),
        }

        file_name = None
        if not visualize:
            date_str = time.strftime("%Y-%m-%d %H:%M:%S")
            file_name = f"Run_{label}_{date_str}"

        metadata = {**self.base_metadata, **(extra_metadata or {})}
        visualize_multiple_results(
            simulation_dict,
            metadata=metadata,
            cost_trajectories=cost_trajectories,
            baseline_cost=base_cost,
            ranks_dict=ranks_dict,
            save_to=file_name,
            dt=self.dt,
        )


    def run_pure(self, visualize: bool = True):
        self.run_comparison("Pure", noise_settings=None, visualize=visualize)

    def run_parametric(self, noise_std: float, time_invariant: bool = False, visualize: bool = True):
        parametric_noise = ParametricNoiseSettings(noise_std=noise_std, time_invariant=time_invariant)
        noise_settings = LQRNoiseSettings(parametric_noise=parametric_noise)
        extra_metadata = {"Parametric Noise Std": noise_std}
        self.run_comparison(
            "Parametric", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_control(self, noise_std: float, visualize: bool = True):
        control_noise = ControlNoiseSettings(noise_std=noise_std)
        noise_settings = LQRNoiseSettings(control_noise=control_noise)
        extra_metadata = {"Control Noise Std": noise_std}
        self.run_comparison(
            "Control", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_measurement(self, noise_std: float, visualize: bool = True):
        measurement_noise = MeasurementNoiseSettings(noise_std=noise_std)
        noise_settings = LQRNoiseSettings(measurement_noise=measurement_noise)
        extra_metadata = {"Measurement Noise Std": noise_std}
        self.run_comparison(
            "Measurement", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_disturbance(self, noise_std: float, disturbance_type: str = "gaussian", visualize: bool = True):
        disturbance_noise = DisturbanceNoiseSettings(noise_std=noise_std, disturbance_type=disturbance_type)
        noise_settings = LQRNoiseSettings(disturbance_noise=disturbance_noise)
        extra_metadata = {
            "Disturbance Noise Std": noise_std,
            "Disturbance Noise Type": disturbance_type,
        }
        self.run_comparison(
            "Disturbance", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )


if __name__ == "__main__":
    # Double Integrator example implementation

    # x" = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    # Position, velocity
    x_0 = np.array([10, 0])

    INFLATED_INPUT_DIM = 10
    INFLATED_OUTPUT_DIM = 5

    model = StateSpaceModel(A, B, x_0)

    # Write identity matrices out, to easily modify them
    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    U_1, _ = np.linalg.qr(np.random.randn(INFLATED_INPUT_DIM, model.A.shape[0]))
    U_2_full, _ = np.linalg.qr(np.random.randn(INFLATED_OUTPUT_DIM, model.B.shape[1]))
    U_2 = U_2_full.T

    #U_1 = np.eye(2)
    #U_2 = np.eye(1)

    # SHOW: Construct a full rank stable K_0
    K_start = np.array([[0.1, 0.5]])
    K_low_rank = np.linalg.pinv(U_2) @ K_start @ np.linalg.pinv(U_1)

    G = np.random.randn(INFLATED_OUTPUT_DIM, INFLATED_INPUT_DIM)

    # Create matrix N such that U_2 @ N @ U_1 = 0
    N = G - np.linalg.pinv(U_2) @ (U_2 @ G @ U_1) @ np.linalg.pinv(U_1)

    # Normaliza N, such that we can control size of the full-rank component
    N = N / np.linalg.norm(N, ord='fro')

    ALPHA = 0.5
    K_0 = K_low_rank + ALPHA * N

    #K_0 = K_start


    LIMIT = 500
    GRADIENT_LR = 1e-2
    TRAJECTORIES = 100
    ROLLOUT_LENGTH = 200
    SMOOTHING_PARAM = 0.5
    STEPS = 2500

    mf_nat_settings = ModelFreePolicyGradientSettings(
        K_0=K_0,
        m=TRAJECTORIES,  # Number of trajectories
        l=ROLLOUT_LENGTH,  # Length of steps in each trajectory
        r=SMOOTHING_PARAM,  # Smoothing parameter
        gradient_steps=LIMIT,
        gradient_lr=GRADIENT_LR
    )

    mf_structured_settings = MFStructuredPolicySettings(
        K_0=K_0,
        beta=0.5,
        lambda_=3000,
        epsilon=1e-3,
        limit=LIMIT,
        m=TRAJECTORIES,
        l=ROLLOUT_LENGTH,
        r=SMOOTHING_PARAM,
        max_linesearch_iter=10,
        stepsize=3.7e-5
    )

    experiment = InflationExperiment(
        model=model,
        Q=Q,
        R=R,
        U_1=U_1,
        U_2=U_2,
        K_0=K_0,
        inflated_input_dim=INFLATED_INPUT_DIM,
        inflated_output_dim=INFLATED_OUTPUT_DIM,
        nat_settings=mf_nat_settings,
        structured_settings=mf_structured_settings,
        dt=0.01,
        sim_steps=STEPS,
    )

    VISUALIZE = True

    try:
        experiment.run_pure(VISUALIZE)
    except Exception as e:
        print(f"Error during pure run: {e}")

    try:
        experiment.run_parametric(noise_std=0.001, time_invariant=False, visualize=VISUALIZE)
    except Exception as e:
        print(f"Error during parametric noise run: {e}")
    
    try:
        experiment.run_control(noise_std=0.75, visualize=VISUALIZE)
    except Exception as e:
        print(f"Error during control noise run: {e}")

    try:
        experiment.run_measurement(noise_std=0.01, visualize=VISUALIZE)
    except Exception as e:
        print(f"Error during measurement noise run: {e}")

    try:
        experiment.run_disturbance(noise_std=0.01, disturbance_type="gaussian", visualize=VISUALIZE)
    except Exception as e:
        print(f"Error during disturbance noise run: {e}")

