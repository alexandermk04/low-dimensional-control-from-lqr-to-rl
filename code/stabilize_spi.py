import numpy as np
import time

from visualization import visualize_spi

from simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    true_cost,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from inflated_lqr import LQRInflated

from structured_policy_iteration import (
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

        self.structured_settings = structured_settings

    def run_comparison(
        self,
        label: str,
        visualize: bool = True,
    ):
        
        base_lqr = LQRDiscrete(
            self.model, self.Q, self.R, dt=self.dt
        )
        lqr = LQRInflated(
            self.model, self.Q, self.R, self.U_1, self.U_2,
            dt=self.dt,
        )

        base_k, base_cost = baseline_solution(base_lqr)
        print(f"[{label}] Baseline solution done")
        low_ks, _ = structured_policy_iteration_mf(lqr=lqr, settings=self.structured_settings)
        print(f"[{label}] Low-rank calculation done")

        cost_list = [true_cost(lqr, self.U_2 @ K @ self.U_1) for K in low_ks]

        rank_list = [np.linalg.matrix_rank(K) for K in low_ks]

        file_name = None
        if not visualize:
            date_str = time.strftime("%Y-%m-%d %H:%M:%S")
            file_name = f"Run_{label}_{date_str}"

        visualize_spi(
            k_ranks=rank_list,
            cost_list=cost_list,
            baseline_cost=base_cost,
            save_to=file_name
        )


    def run_pure(self, visualize: bool = True):
        self.run_comparison("Pure", noise_settings=None, visualize=visualize)
 


if __name__ == "__main__":
    # Double Integrator example implementation

    # x" = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    # Position, velocity
    x_0 = np.array([10, 0])

    INFLATED_INPUT_DIM = 2
    INFLATED_OUTPUT_DIM = 1

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

    #ALPHA = 0.5
    #K_0 = K_low_rank + ALPHA * N

    K_0 = K_start

    LIMIT = 3000
    TRAJECTORIES = 100
    ROLLOUT_LENGTH = 200
    SMOOTHING_PARAM = 0.5

    # ------ Important parameters ------
    BETA = 0.5
    LAMBDA = 3000
    # -------------

    mf_structured_settings = MFStructuredPolicySettings(
        K_0=K_0,
        beta=0.5,
        lambda_=100,
        epsilon=1e-3,
        limit=LIMIT,
        m=TRAJECTORIES,
        l=ROLLOUT_LENGTH,
        r=SMOOTHING_PARAM
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
        structured_settings=mf_structured_settings,
        dt=0.01,
    )

    VISUALIZE = True

    try:
        experiment.run_pure(VISUALIZE)
    except Exception as e:
        print(f"Error during pure run: {e}")

