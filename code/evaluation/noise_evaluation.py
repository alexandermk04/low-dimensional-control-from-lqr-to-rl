"""Compare the influence of different noise types on the model-free natural policy gradient.

Run it from ``code/`` with::

    python -m evaluation.noise_evaluation
"""

import numpy as np
import time

from visualization import visualize_multiple_results

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

from lqr_solvers.mf_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_natural_policy_gradient
)


class NoiseExperiment:
    """One LQR experiment with the possibility to vary noise conditions

    Parameters
    ----------
    model : StateSpaceModel
        Continuous-time plant.
    Q, R : np.ndarray
        Cost weights on the state and the input.
    K_0 : np.ndarray
        Initial gain of shape ``(m, n)``.
    nat_settings : ModelFreePolicyGradientSettings
        Hyperparameters for the natural policy gradient.
    max_initial_state : np.ndarray, optional
        Bound on the sampled initial state. Defaults to ``[10, 1]``.
    dt : float, optional
        Sampling interval. Defaults to ``0.01``.
    sim_steps : int, optional
        Length of the final trajectory rollouts. Defaults to ``2500``.

    Attributes
    ----------
    base_metadata : dict
        Hyperparameters of a run rendered into every figure's text panel.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        Q: np.ndarray,
        R: np.ndarray,
        K_0: np.ndarray,
        nat_settings: ModelFreePolicyGradientSettings,
        max_initial_state: np.ndarray | None = None,
        dt: float = 0.01,
        sim_steps: int = 2500,
    ):
        self.model = model
        self.Q = Q
        self.R = R
        self.dt = dt
        self.sim_steps = sim_steps

        self.K_0 = K_0
        self.nat_settings = nat_settings

        if max_initial_state is None:
            max_initial_state = np.array([10, 1])
        self.max_initial_state = max_initial_state

        self.base_metadata = {
            "Gradient Steps": nat_settings.gradient_steps,
            "Gradient LR": nat_settings.gradient_lr,
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
        """Solve the problem under the given noise settings and plot the result.

        Parameters
        ----------
        label : str
            Short name for the noise condition, used in legends and in the saved
            filename.
        noise_settings : LQRNoiseSettings or None, optional
            Noise to expose the system to. ``None`` runs clean.
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

        settings = {
            "model": self.model,
            "Q": self.Q,
            "R": self.R,
            "max_initial_state": self.max_initial_state,
            "resample_initial_state": False,
            "dt": self.dt,
        }

        base_lqr = LQRDiscrete(
            settings=LQRDiscreteSettings(**settings, noise_settings=None)
        )

        lqr = LQRDiscrete(
            settings=LQRDiscreteSettings(**settings, noise_settings=noise_settings)
        )

        base_lqr.reset()
        lqr.reset()

        base_k, base_cost = baseline_solution(base_lqr)
        print(f"[{label}] Baseline solution done")
        ks, _ = model_free_natural_policy_gradient(lqr=lqr, settings=self.nat_settings)
        print(f"[{label}] Natural policy gradient done")

        cost_trajectories = {
            f"Natural Gradient {label}": [true_cost(lqr, K) for K in ks],
        }

        simulation_dict = {
            f"Baseline {label}": simulate(lqr=base_lqr, K=base_k, n_steps=self.sim_steps),
            f"Natural Gradient {label}": simulate(lqr=lqr, K=ks[-1], n_steps=self.sim_steps),
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
            save_to=file_name,
            dt=self.dt,
        )

    def run_pure(self, visualize: bool = True):
        """Run with no noise.

        Parameters
        ----------
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        self.run_comparison("Pure", noise_settings=None, visualize=visualize)

    def run_parametric(self, noise_std: float, time_invariant: bool = False, visualize: bool = True):
        """Run with parametric noise applied..

        Parameters
        ----------
        noise_std : float
            Standard deviation of the perturbation on ``A`` and ``B``.
        time_invariant : bool, optional
            ``False`` (default) redraws every step; ``True`` freezes one
            perturbation for the whole run.
        visualize : bool, optional
            See :meth:`run_comparison`.
        """
        parametric_noise = ParametricNoiseSettings(noise_std=noise_std, time_invariant=time_invariant)
        noise_settings = LQRNoiseSettings(parametric_noise=parametric_noise)
        extra_metadata = {
            "Parametric Noise Std": noise_std,
            "Parametric Time Invariant": time_invariant,
        }
        self.run_comparison(
            "Parametric", noise_settings=noise_settings,
            extra_metadata=extra_metadata, visualize=visualize,
        )

    def run_control(self, noise_std: float, visualize: bool = True):
        """Run with multiplicative noise applied to the control input.

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
        """Run with added measurement noise.

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
        """Run with an external disturbance acting on the plant.

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

    # x' = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])

    model = StateSpaceModel(A, B)

    # Write identity matrices out, to easily modify them

    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    K_0 = np.array([[0.1, 0.5]])

    GRADIENT_STEPS = 200
    TRAJECTORIES = 100
    ROLLOUT_LENGTH = 200
    SMOOTHING_PARAM = 0.01
    STEPS = 2500

    mf_nat_settings = ModelFreePolicyGradientSettings(
        K_0=K_0,
        m=TRAJECTORIES,  # Number of trajectories
        l=ROLLOUT_LENGTH,  # Length of steps in each trajectory
        r=SMOOTHING_PARAM,  # Smoothing parameter
        gradient_steps=GRADIENT_STEPS,
        gradient_lr=0.001
    )

    experiment = NoiseExperiment(
        model=model,
        Q=Q,
        R=R,
        K_0=K_0,
        nat_settings=mf_nat_settings,
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
        experiment.run_control(noise_std=0.1, visualize=VISUALIZE)
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
