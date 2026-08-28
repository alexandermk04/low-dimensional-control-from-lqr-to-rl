"""Full-rank versus low-rank control in an inflated observation/action space.

Run it from ``code/`` with::

    python -m evaluation.inflate_evaluation
"""

import numpy as np
import time

from seeding import SEED, create_seed_sequences, rng
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
    LQRDiscreteSettings,
    simulate,
    true_cost,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from lqr_base.inflated_lqr import LQRInflated, LQRInflatedSettings

from lqr_solvers.mf_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_natural_policy_gradient
)

from lqr_solvers.structured_policy_iteration import (
    MFStructuredPolicySettings,
    structured_policy_iteration_mf
)


class InflationExperiment:
    """One inflated problem, runnable under each noise channel in turn.

    Parameters
    ----------
    model : StateSpaceModel
        Continuous-time plant.
    Q, R : np.ndarray
        Cost weights on the true state and the true input.
    structured_settings : MFStructuredPolicySettings
        Hyperparameters for the low-rank solver.
    dt : float, optional
        Sampling interval. Defaults to ``0.01``.
    sim_steps : int, optional
        Length of the final trajectory rollouts. Defaults to ``2500``.

    Attributes
    ----------
    base_metadata : dict
        Hyperparameters rendered into every figure's text panel.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        Q: np.ndarray,
        R: np.ndarray,
        inflated_input_dim: int,
        inflated_output_dim: int,
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

        self.base_metadata = {
            "Inflated Input Dim": inflated_input_dim,
            "Inflated Output Dim": inflated_output_dim,
            "Simulation Steps": sim_steps,
            "Seed": SEED,
        }

    def run_comparison(
        self,
        label: str,
        noise_settings: LQRNoiseSettings | None = None,
        extra_metadata: dict | None = None,
        visualize: bool = True,
    ):
        """Run baseline, full-rank and low-rank under one noise condition.

        Parameters
        ----------
        label : str
            Short name for this condition, used in legends and in the saved
            filename.
        noise_settings : LQRNoiseSettings or None, optional
            Noise to expose both the baseline and the inflated system to.
            ``None`` runs clean.
        extra_metadata : dict or None, optional
            Condition-specific entries merged over ``base_metadata`` for the
            figure's text panel.
        visualize : bool, optional
            ``True`` shows the figure; ``False`` saves it as
            ``Run_<label>_<timestamp>.png``.

        Returns
        -------
        None
            Shows or saves the six-panel comparison figure.
        """

        base_seeds = create_seed_sequences(["base"])

        base_settings = LQRDiscreteSettings(
            model=self.model,
            Q=self.Q,
            R=self.R,
            max_initial_state=np.array([10, 1]),
            resample_initial_state=False,
            noise_settings=noise_settings,
            seed_seq=base_seeds["base"],
            dt=self.dt
        )
        
        base_lqr = LQRDiscrete(
            settings=base_settings
        )

        seeds = create_seed_sequences(["lqr", "matrices", "mf_nat", "mf_spi"])

        mat_rng = rng(seeds["matrices"])

        U_1, _ = np.linalg.qr(mat_rng.standard_normal((INFLATED_INPUT_DIM, model.A.shape[0])))
        U_2_full, _ = np.linalg.qr(mat_rng.standard_normal((INFLATED_OUTPUT_DIM, model.B.shape[1])))
        U_2 = U_2_full.T
    
        # Construct a full rank stable K_0
        K_start = np.array([[0.1, 0.5]])
        K_low_rank = np.linalg.pinv(U_2) @ K_start @ np.linalg.pinv(U_1)
    
        G = mat_rng.standard_normal((INFLATED_OUTPUT_DIM, INFLATED_INPUT_DIM))
    
        # Create matrix N such that U_2 @ N @ U_1 = 0
        N = G - np.linalg.pinv(U_2) @ (U_2 @ G @ U_1) @ np.linalg.pinv(U_1)
    
        # Normaliza N, such that we can control size of the full-rank component
        N = N / np.linalg.norm(N, ord='fro')
    
        ALPHA = 0.5
        K_0 = K_low_rank + ALPHA * N
    
        limit = 500
        gradient_lr = 1e-2
        trajectories = 100
        rollout_length = 200
        smoothing_param = 0.5
    
        mf_nat_settings = ModelFreePolicyGradientSettings(
            K_0=K_0,
            m=trajectories,  # Number of trajectories
            l=rollout_length,  # Length of steps in each trajectory
            r=smoothing_param,  # Smoothing parameter
            gradient_steps=limit,
            gradient_lr=gradient_lr,
            seed_seq=seeds["mf_nat"]
        )

        beta = 0.5
        lambda_ = 3000
        epsilon = 1e-3
        stepsize = 3.7e-5
        max_linesearch_iter = 10
    
        mf_structured_settings = MFStructuredPolicySettings(
            K_0=K_0,
            beta=beta,
            lambda_=lambda_,
            epsilon=epsilon,
            limit=limit,
            m=trajectories,
            l=rollout_length,
            r=smoothing_param,
            max_linesearch_iter=max_linesearch_iter,
            stepsize=stepsize,
            seed_seq=seeds["mf_spi"]
        )


        lqr_settings = LQRInflatedSettings(
            model=self.model,
            Q=self.Q,
            R=self.R,
            max_initial_state=np.array([10, 1]),
            resample_initial_state=False,
            noise_settings=noise_settings,
            seed_seq=seeds["lqr"],
            dt=self.dt,
            U_1=U_1,
            U_2=U_2
        )

        lqr = LQRInflated(
            settings=lqr_settings
        )

        settings_metadata = {
            "Gradient Steps": limit,
            "Gradient LR": gradient_lr,
            "Lambda SPI": lambda_,
            "Beta SPI": beta,
            "Trajectories per Gradient Step": trajectories,
            "Rollout Length": rollout_length,
            "Smoothing Parameter": smoothing_param,
        }

        base_k, base_cost = baseline_solution(base_lqr)
        print(f"[{label}] Baseline solution done")
        print(f"[{label}] Baseline cost: {base_cost:.3e}")
        
        full_ks, _ = model_free_natural_policy_gradient(lqr=lqr, settings=mf_nat_settings)
        print(f"[{label}] Full-rank calculation done")

        lqr.reset_seeding()
        low_ks, _ = structured_policy_iteration_mf(lqr=lqr, settings=mf_structured_settings)
        print(f"[{label}] Low-rank calculation done")

        cost_trajectories = {
            'Full-rank': [true_cost(lqr, lqr.effective_gain(K)) for K in full_ks],
            'Low-rank': [true_cost(lqr, lqr.effective_gain(K)) for K in low_ks],
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
            file_name = f"Run_{label}_{date_str}_seed_{SEED}"

        metadata = {**self.base_metadata, **settings_metadata, **(extra_metadata or {})}
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
        """Run the comparison with no noise.

        Parameters
        ----------
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        self.run_comparison("Pure", noise_settings=None, visualize=visualize)

    def run_parametric(self, noise_std: float, time_invariant: bool = False, visualize: bool = True):
        """Run the comparison with parametric perturbation.

        Parameters
        ----------
        noise_std : float
            Standard deviation of the perturbation on ``A`` and ``B``.
        time_invariant : bool, optional
            ``False`` (default) redraws every step, ``True`` freezes one perturbation.
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        parametric_noise = ParametricNoiseSettings(noise_std=noise_std, time_invariant=time_invariant)
        noise_settings = LQRNoiseSettings(parametric_noise=parametric_noise)
        extra_metadata = {"Parametric Noise Std": noise_std}
        self.run_comparison(
            "Parametric", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_control(self, noise_std: float, visualize: bool = True):
        """Run the comparison with multiplicative control noise.

        Parameters
        ----------
        noise_std : float
            Relative error on the commanded input.
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        control_noise = ControlNoiseSettings(noise_std=noise_std)
        noise_settings = LQRNoiseSettings(control_noise=control_noise)
        extra_metadata = {"Control Noise Std": noise_std}
        self.run_comparison(
            "Control", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_measurement(self, noise_std: float, visualize: bool = True):
        """Run the comparison with additive sensor noise.

        Parameters
        ----------
        noise_std : float
            Standard deviation added to each observation entry.
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        measurement_noise = MeasurementNoiseSettings(noise_std=noise_std)
        noise_settings = LQRNoiseSettings(measurement_noise=measurement_noise)
        extra_metadata = {"Measurement Noise Std": noise_std}
        self.run_comparison(
            "Measurement", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_disturbance(self, noise_std: float, disturbance_type: str = "gaussian", visualize: bool = True):
        """Run the comparison with an external disturbance acting on the plant.

        Parameters
        ----------
        noise_std : float
            Magnitude of the disturbance on the successor state.
        disturbance_type : {"gaussian", "sinusoidal", "constant"}, optional
            See :class:`~lqr_base.noise.DisturbanceNoiseSettings`.
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
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

    INFLATED_INPUT_DIM = 50
    INFLATED_OUTPUT_DIM = 25

    model = StateSpaceModel(A, B)

    # Write identity matrices out, to easily modify them
    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    STEPS = 2500

    experiment = InflationExperiment(
        model=model,
        Q=Q,
        R=R,
        inflated_input_dim=INFLATED_INPUT_DIM,
        inflated_output_dim=INFLATED_OUTPUT_DIM,
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

