"""
Compares the S-PI variants against the baseline LQR solution.

Run it from ``code/`` with::

    python -m evaluation.spi_evaluation
"""

import numpy as np
import time

from seeding import create_seed_sequences, rng
from settings import TEST_SEED
from config import (
    make_lifts,
    make_full_rank_K0,
    double_integrator_model,
    default_costs,
    DEFAULT_DT,
    DEFAULT_MAX_INITIAL_STATE,
    DEFAULT_SIM_STEPS
)

from visualization import visualize_spi

from lqr_base.simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    LQRDiscreteSettings,
    simulate,
    true_cost,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from lqr_base.inflated_lqr import LQRInflated, LQRInflatedSettings

from lqr_solvers.structured_policy_iteration import (
    StructuredPolicySettings,
    structured_policy_iteration,
    MFStructuredPolicySettings,
    structured_policy_iteration_mf
)


class SPIInflationExperiment:
    """S-PI on an inflated LQR system.

    Parameters
    ----------
    model : StateSpaceModel
        Continuous-time plant.
    Q, R : np.ndarray
        Cost weights.
    inflated_input_dim, inflated_output_dim : int
        Dimensions of the lifted spaces.
    dt : float, optional
        Sampling interval. Defaults to ``0.01``.
    sim_steps : int, optional
        Length of the final trajectory rollouts. Defaults to ``2500``.
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

    def run_comparison(
        self,
        label: str,
        visualize: bool = True,
    ):
        """Run S-PI on the inflated system and plot rank against cost.

        Parameters
        ----------
        label : str
            Short name for the condition, used in the saved filename.
        visualize : bool, optional
            ``True`` shows the figure; ``False`` saves it as
            ``Run_<label>_<timestamp>.png``.

        Returns
        -------
        None
            Shows or saves the four-panel S-PI figure.
        """

        settings = {
            "model": self.model,
            "Q": self.Q,
            "R": self.R,
            "max_initial_state": DEFAULT_MAX_INITIAL_STATE,
            "resample_initial_state": False,
            "noise_settings": None,
            "dt": self.dt,
        }

        base_seeds = create_seed_sequences(["base"], TEST_SEED)

        discrete_settings = LQRDiscreteSettings(
            **settings,
            seed_seq=base_seeds["base"]
        )

        base_lqr = LQRDiscrete(
            settings=discrete_settings
        )

        # LQR first to sample same seq as in base
        seeds = create_seed_sequences(["lqr", "matrices", "mf_structured"], TEST_SEED)       

        mat_rng = rng(seeds["matrices"])

        U_1, U_2 = make_lifts(
            self.inflated_input_dim, self.inflated_output_dim, self.model, mat_rng
        )

        inflated_settings = LQRInflatedSettings(
            **settings,
            seed_seq=seeds["lqr"],
            U_1=U_1,
            U_2=U_2
        )

        lqr = LQRInflated(
            settings=inflated_settings
        )
    
        ALPHA = 1.0
        K_0 = make_full_rank_K0(U_1, U_2, mat_rng, alpha=ALPHA)
    
        # Create the settings
        LIMIT = 100
        TRAJECTORIES = 100
        ROLLOUT_LENGTH = 200
        SMOOTHING_PARAM = 0.5
        MAX_LINESEARCH_ITER = 10
    
        # ------ Important parameters ------
        BETA = 0.5
        LAMBDA = 3000
        EPSILON = 1e-3
        STEPSIZE = 3.7e-5
        # -------------
    
        structured_settings = StructuredPolicySettings(
            K_0=K_0,
            beta=BETA,
            lambda_=LAMBDA,
            epsilon=EPSILON,
            limit=LIMIT,
            max_linesearch_iter=MAX_LINESEARCH_ITER
        )
    
        mf_structured_settings = MFStructuredPolicySettings(
            K_0=K_0,
            beta=BETA,
            lambda_=LAMBDA,
            epsilon=EPSILON,
            limit=LIMIT,
            m=TRAJECTORIES,
            l=ROLLOUT_LENGTH,
            r=SMOOTHING_PARAM,
            max_linesearch_iter=MAX_LINESEARCH_ITER,
            stepsize=STEPSIZE,
            seed_seq=seeds["mf_structured"]
        )

        base_k, base_cost = baseline_solution(base_lqr)
        print(f"[{label}] Baseline solution done")

        #ks, _ = structured_policy_iteration(lqr=lqr, settings=structured_settings)
        #print(f"[{label}] Model-based calculation done")
        # lqr.reset_seeding()

        mf_ks, _ = structured_policy_iteration_mf(lqr=lqr, settings=mf_structured_settings)
        print(f"[{label}] Model-free calculation done")

        #true_cost_spi = [true_cost(lqr, U_2 @ K @ U_1) for K in ks]
        true_cost_spi_mf = [true_cost(lqr, U_2 @ K @ U_1) for K in mf_ks]

        cost_dict = {
            #"SPI (Model-Based)": true_cost_spi,
            "SPI (Model-Free)": true_cost_spi_mf,
        }

        #rank_list = [np.linalg.matrix_rank(K) for K in ks]
        rank_list_mf = [np.linalg.matrix_rank(K) for K in mf_ks]

        rank_dict = {
            #"SPI (Model-Based)": rank_list,
            "SPI (Model-Free)": rank_list_mf,
        }

        simulation_dict = {
            "Baseline LQR": simulate(base_lqr, base_k, n_steps=self.sim_steps),
            #"SPI (Model-Based)": simulate(lqr, ks[-1], n_steps=self.sim_steps),
            "SPI (Model-Free)": simulate(lqr, mf_ks[-1], n_steps=self.sim_steps),
        }

        file_name = None
        if not visualize:
            date_str = time.strftime("%Y-%m-%d %H:%M:%S")
            file_name = f"Run_{label}_{date_str}_seed_{TEST_SEED}"

        visualize_spi(
            k_ranks=rank_dict,
            cost_trajectories=cost_dict,
            baseline_cost=base_cost,
            simulation_dict=simulation_dict,
            seed=TEST_SEED,
            dt=self.dt,
            save_to=file_name
        )


    def run_pure(self, visualize: bool = True):
        """Run the noise-free condition.

        Parameters
        ----------
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        self.run_comparison("Pure", visualize=visualize)
 


if __name__ == "__main__":

    INFLATED_INPUT_DIM = 100
    INFLATED_OUTPUT_DIM = 50

    model = double_integrator_model()

    Q, R = default_costs()

    experiment = SPIInflationExperiment(
        model=model,
        Q=Q,
        R=R,
        inflated_input_dim=INFLATED_INPUT_DIM,
        inflated_output_dim=INFLATED_OUTPUT_DIM,
        dt=DEFAULT_SIM_STEPS,
        sim_steps=DEFAULT_SIM_STEPS
    )

    VISUALIZE = True

    experiment.run_pure(VISUALIZE)

