"""Model-free policy gradient descent from rollouts only.

Two variants are provided:

- :func:`model_free_policy_gradient` - plain descent on the estimated
  gradient.
- :func:`model_free_natural_policy_gradient` - the natural gradient, which
  preconditions by the inverse state covariance.

References
----------
.. [1] M. Fazel, R. Ge, S. M. Kakade, M. Mesbahi, "Global Convergence of Policy
       Gradient Methods for the Linear Quadratic Regulator", ICML 2018.
       https://arxiv.org/abs/1801.05039
"""

from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

from seeding import rng

from evaluation.probe import Probe, record_iteration

from lqr_base.simple_lqr import (
    LQRDiscrete,
    true_cost,
)

@dataclass
class ModelFreePolicyGradientSettings:
    """Hyperparameters shared by both model-free variants.

    Attributes
    ----------
    K_0 : np.ndarray
        Initial gain, must be stabilizing.
    m : int
        Number of perturbed trajectories per gradient step.
    l : int
        Steps per rollout. Finite-horizon cost estimate.
    r : float
        Smoothing radius. The perturbations have Frobenius norm exactly ``r``.
    gradient_steps : int
        Number of descent steps.
    gradient_lr : float
        Fixed step size.
    seed_seq : np.random.SeedSequence
        Seed sequence for reproducibility.
    probe : evaluation.probe.Probe or None, optional
        Instrumentation recording cost, sample budget and total time per
        iteration. Defaults to ``None``
    """

    K_0: np.ndarray  # Initial policy
    m: int  # Number of trajectories
    l: int  # Length of steps in each trajectory
    r: float  # Smoothing parameter
    gradient_steps: int  # Number of gradient steps
    gradient_lr: float  # Gradient step size
    seed_seq: np.random.SeedSequence  # Seed sequence for reproducibility
    probe: Probe | None = None  # Optional per-iteration instrumentation

# Observation norm beyond which a rollout is treated as diverged.
DIVERGENCE_THRESHOLD = 1e5
# Cost charged to a diverged rollout
DIVERGENCE_PENALTY = 1e10


def run_trajectory(lqr: LQRDiscrete, K: np.ndarray, l: int) -> tuple[np.float64, np.float64]:
    """Run a trajectory once and accumulate cost and covariance.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to step. Must already be set up.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.
    l : int
        Number of steps.

    Returns
    -------
    total_cost : float
        Sum of the stage costs over the rollout.
    sigma : np.ndarray
        ``sum_t y_t y_t'``, of shape ``(p, p)``: the unnormalized empirical
        covariance of the observations, which is what the natural gradient
        preconditions by.
    """
    y_traj = []
    v_traj = []
    costs = []
    sigma = np.zeros((K.shape[1], K.shape[1]))

    total_cost = 0.0

    for _ in range(l):
        res = lqr.next_state(K)

        diverged = (
            not np.isfinite(res.cost)
            or not np.all(np.isfinite(res.y_t))
            or np.linalg.norm(res.y_t) > DIVERGENCE_THRESHOLD
        )

        if diverged:
            total_cost = sum(costs) + DIVERGENCE_PENALTY
            return total_cost, sigma

        y_traj.append(res.y_t)
        v_traj.append(res.v_t)
        costs.append(res.cost)
        sigma += np.outer(res.y_t, res.y_t)

    return sum(costs), sigma

def estimate_gradient(lqr: LQRDiscrete, K: np.ndarray, m: int, l: int, r: float, rngen: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate the policy gradient by random search around ``K``.

    Draws ``m`` random perturbations ``U`` of fixed Frobenius norm ``r``, and
    for each one evaluates both ``K + U`` and ``K - U`` (two point estimation). Correlating the
    resulting costs with the perturbations gives an estimate of the gradient.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to sample from. Reset and seeded with noise before each rollout.
    K : np.ndarray
        Point at which to estimate the gradient, of shape ``(q, p)``.
    m : int
        Number of perturbation pairs.
    l : int
        Steps per rollout.
    r : float
        Perturbation radius in Frobenius norm.
    rngen : np.random.Generator
        The seeded generator for reproducibility

    Returns
    -------
    cost_gradient_estimate : np.ndarray
        Estimated gradient, same shape as ``K``.
    sigma_estimate : np.ndarray
        Mean observation covariance over all rollouts, of shape ``(p, p)``.
        Used as the natural-gradient preconditioner.
    mean_cost : float
        Mean cost over the perturbed rollouts. Following [1]_ section D.3 this
        -- not the cost at ``K`` itself -- is the cost associated with the
        current iterate, since it is the quantity the estimator is actually
        unbiased for.

    Notes
    -----
    Three implementation details carry most of the estimator's quality, all
    recorded as hard-won in ``code/notes/PROCESS.md``:

    - **Two-point estimation.** Evaluating ``K + U`` and ``K - U`` cuts the variance
      drastically.
    - **Common random numbers.** The RNG state is captured before the ``K + U``
      rollout and restored before the ``K - U`` one, so both see *identical*
      noise and initial conditions. The difference between them is then
      attributable to the perturbation rather than to the sampling.
      TODO: This is not realistic in practice, so should maybe be removed
    - **Mean centering.** Subtracting the mean cost before correlating removes
      the common offset, which would otherwise contribute pure variance.

    Warnings
    --------
    The return annotation says ``np.ndarray`` but the function returns a
    three-tuple.
    """
    pertubations = []
    cost_differences = []
    cost_estimates = []
    sigma_estimates = []

    d = K.shape[0] * K.shape[1]

    for _ in range(m):

        # Sample a random matrix with K's shape
        U = rngen.standard_normal(K.shape)

        # Normalize to have unit frobenius norm
        U /= np.linalg.norm(U, ord='fro')
        U *= r

        pertubations.append(U)

        # SHOW: Two point estimator
        K_i_plus = K + U
        K_i_minus = K - U

        state_rng_state = lqr.state_rng.bit_generator.state
        lqr.reset()
        cost_estimate_plus, sigma_estimate_plus = run_trajectory(lqr, K_i_plus, l)

        # Use same initial state
        lqr.state_rng.bit_generator.state = state_rng_state
        lqr.reset()
        cost_estimate_minus, sigma_estimate_minus = run_trajectory(lqr, K_i_minus, l)

        # Calculate values for this perturbation pair
        cost_differences.append(cost_estimate_plus - cost_estimate_minus)

        overall_cost_estimate = (cost_estimate_plus + cost_estimate_minus)/2
        cost_estimates.append(overall_cost_estimate)

        sigma_estimates.append(sigma_estimate_plus + sigma_estimate_minus)

    # Centering the cost differences by subtracting the mean to reduce variance
    mean_cost_diffs = np.mean(cost_differences)
    centered_cost_diffs = cost_differences - mean_cost_diffs

    mean_cost = np.mean(cost_estimates)

    cost_gradient_estimate = sum(1/2 * (d / (r ** 2)) * C * U for C, U in zip(centered_cost_diffs, pertubations))
    cost_gradient_estimate /= m

    sigma_estimate = sum(sig for sig in sigma_estimates) / m

    return cost_gradient_estimate, sigma_estimate, mean_cost

def model_free_policy_gradient(lqr: LQRDiscrete, settings: ModelFreePolicyGradientSettings):
    """Gradient descend with sampled estimates.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over. 
    settings : ModelFreePolicyGradientSettings
        Settings for this algorithm.

    Returns
    -------
    ks : list[np.ndarray]
        Every iterate including ``K_0``, of length ``gradient_steps + 1``.
    costs : list[float]
        Estimated cost at each step, of length ``gradient_steps``. 
    """
    lqr.reset()
    K = settings.K_0

    rngen = rng(settings.seed_seq)

    costs = []
    ks = [K]

    progress_bar = tqdm(range(settings.gradient_steps), desc="Model-Free Policy Gradient", unit="step")

    for i in progress_bar:
        cost_gradient_estimate, sigma_estimate, mean_cost = estimate_gradient(lqr, K, settings.m, settings.l, settings.r, rngen)

        # By section D.3, above lemma 29, the expected cost of the pertubed policy is used for the cost estimate
        costs.append(mean_cost)

        K = K - settings.gradient_lr * cost_gradient_estimate

        ks.append(K)

        record_iteration(settings, lqr, progress_bar, i, K, cost_estimate=mean_cost)

    return ks, costs

def model_free_natural_policy_gradient(lqr: LQRDiscrete, settings: ModelFreePolicyGradientSettings):
    """Descend on the gradient preconditioned by the inverse state covariance.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over.
    settings : ModelFreePolicyGradientSettings
        Settings for this algorithm.

    Returns
    -------
    ks : list[np.ndarray]
        Every iterate including ``K_0``.
    costs : list[float]
        Estimated cost at each step; see :func:`model_free_policy_gradient`.

    Notes
    -----
    The preconditioner is a truncated pseudoinverse,
    ``pinv(sigma, rcond=1e-6)``, not a plain inverse. With an inflated
    observation space ``sigma`` is rank-deficient by construction - the
    observations live on an ``n``-dimensional subspace of a ``p``-dimensional
    space - so a plain inverse either fails or amplifies numerical noise into
    an enormous step. This was one of the crucial changes to
    fix inflated runs into running stably.
    """
    lqr.reset()
    K = settings.K_0

    rngen = rng(settings.seed_seq)

    costs = []
    ks = [K]

    progress_bar = tqdm(range(settings.gradient_steps), desc="Model-Free Natural Policy Gradient", unit="step")

    for i in progress_bar:
        cost_gradient_estimate, sigma_estimate, mean_cost = estimate_gradient(lqr, K, settings.m, settings.l, settings.r, rngen)

        # By section D.3, above lemma 29, the expected cost of the pertubed policy is used for the cost estimate
        costs.append(mean_cost)

        # SHOW: Use a pseudoinverse with truncation to remain stable for inflated inputs
        K = K - settings.gradient_lr * (cost_gradient_estimate @ np.linalg.pinv(sigma_estimate, rcond=1e-6))

        ks.append(K)

        record_iteration(settings, lqr, progress_bar, i, K, cost_estimate=mean_cost)

    return ks, costs
