"""
The single file that holds the set parameters for all experiments.
"""

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
# Were drawn randomly, now fixed for reproducibility. Tests are averaged over these
SEEDS = (836, 16570, 27491, 46125, 60177, 66059, 68719, 69486, 73760, 76464)

# Seed used for simulating trajectories, such that start state is identical
SHARED_SEED = 33843

# Seed that was used for past evaluations
TEST_SEED = 2026

# ---------------------------------------------------------------------------
# Test protocol
# ---------------------------------------------------------------------------
# Each seed is run for this many episodes, each of this many steps.
TEST_EPISODES = 100
TEST_HORIZON = 2000

# Length of the trajectories drawn in the figures.
FIGURE_HORIZON = 800

# Rollout length of the mid-training Monte Carlo probe for non-linear RL policies.
PROBE_HORIZON = 800

# Episodes to evaluate during training run
PROBE_EPISODES = 100

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Safety cap on iterations for the LQR solvers
SOLVER_MAX_ITERATIONS = 20_000

# Size of the full-rank component of the initial gain on an inflated system.
K0_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Sample sizes (appendix)
# ---------------------------------------------------------------------------
# l = rollout lengths, m = trajectory counts.
ROLLOUT_LENGTHS = (200, 500, 800)
TRAJECTORY_COUNTS = (100, 200)
SAMPLE_GRID = tuple((l, m) for l in ROLLOUT_LENGTHS for m in TRAJECTORY_COUNTS)

# Per inflation size sample sizes for model-free approach 

# Filled from `python -m experiments.appendix_sample_size`; values are (l, m).
CHOSEN_SAMPLE_CELL = {
    "base": (800, 100),
    "small": (800, 100),
    "medium": (800, 100),
    "large": (800, 100),
}

# Length of excitation rollouts
IDENTIFICATION_LENGTHS = (500, 1000, 2000, 4000, 8000)

# Past/future horizon of the block Hankel matrices for N4SID
IDENTIFICATION_HORIZONS = (5, 10, 20)

# Filled from `python -m experiments.appendix_identification_length`;
# values are (trajectory_length, hankel_horizon).
CHOSEN_IDENTIFICATION = {
    "base": (500, 5),
    "small": (500, 5),
    "medium": (500, 5),
    "large": (2000, 5),
}

SUBOPTIMAL_IDENTIFICATION = {
    "medium": (500, 10),
    "large": (1000, 5),
}

# Rollout length for training
TRAIN_HORIZON = 800

# Safety cap on environment steps for the RL solvers
RL_MAX_TIMESTEPS = 10_000_000

# Environment steps between learning-curve samples
RL_EVAL_FREQ = 25_000

# Convergence stopping rule. Stop once the moving mean over ``window`` updates 
# is not ``rel_tol`` lower than ``patience`` updates ago.
CONVERGENCE = {
    "model_based": dict(rel_tol=1e-4, patience=50, window=1),
    "model_free": dict(rel_tol=1e-3, patience=100, window=50),
    "rl": dict(rel_tol=1e-2, patience=10, window=3),
}


# ---------------------------------------------------------------------------
# Noise, applied at test time (chapter 4)
# ---------------------------------------------------------------------------
NOISE_LEVELS = ("none", "weak", "standard", "strong", "divergent")
NOISE_CHANNELS = ("measurement", "control", "parametric", "disturbance")

# Increase of the baseline controller test cost on the base system
NOISE_COST_TARGETS = {"weak": 1.2, "standard": 1.5, "strong": 3.0, "divergent": 10.0}

# Standard deviation per channel and level. ``none`` must stay 0.0.
# Filled from `python -m experiments.appendix_calibrate_noise`.

NOISE_STD = {
    "measurement": {
        "none": 0.0,
        "weak": 0.389,
        "standard": 0.607,
        "strong": 1.22,
        "divergent": 2.58,
    },
    "control": {
        "none": 0.0,
        "weak": 1.07,
        "standard": 1.67,
        "strong": 3.19,
        "divergent": 5.68,
    },
    "parametric": {
        "none": 0.0,
        "weak": 0.0211,
        "standard": 0.0304,
        "strong": 0.0451,
        "divergent": 0.0582,
    },
    "disturbance": {
        "none": 0.0,
        "weak": 0.00767,
        "standard": 0.0122,
        "strong": 0.0246,
        "divergent": 0.0519,
    },
}

# Shape of the two channels that have a mode besides their intensity.
PARAMETRIC_TIME_INVARIANT = False
DISTURBANCE_TYPE = "sinusoidal"  # "gaussian" | "sinusoidal" | "constant"

# The single level the per-channel comparison (table 4, plot 5) is drawn at.
# TODO: set after inspecting the results of `python -m experiments.ch4_noise`
CHOSEN_NOISE_LEVEL = "standard"

# The single channel the per-level comparison (table 3, plot 4) is drawn for.
# TODO: set after inspecting the results of `python -m experiments.ch4_noise`
CHOSEN_NOISE_CHANNEL = "measurement"


# ---------------------------------------------------------------------------
# Inflation (chapter 5)
# ---------------------------------------------------------------------------
INFLATION_SIZES = {
    "base": (2, 1), # use the inflation style with the base dimensions
    "small": (10, 5),
    "medium": (50, 25),
    "large": (200, 100),
    "pixel": (18432, 256), # size for chapter 6 experiments
}

# Which sizes each chapter looks at
CH4_INFLATION_SIZES = ("base", "small", "medium", "large")

# The chapter-4 noise grid setting
CH4_REFERENCE_INFLATION = "large"

# Chapter 5 different inflation sizes
CH5_INFLATION_SIZES = ("small", "medium", "large")

# The size chapter 5's noise setting
CH5_REFERENCE_INFLATION = "large"

CH5_PIXEL_INFLATION = "pixel"


# ---------------------------------------------------------------------------
# Appendix: model-free S-PI hyperparameters
# ---------------------------------------------------------------------------
SPI_MF_LAMBDAS = (4e2, 4e3, 4e4, 4e5, 4e6)
SPI_MF_STEPSIZES = (3.5e-9, 3.5e-8, 3.5e-7, 3.5e-6, 3.5e-5)


# ---------------------------------------------------------------------------
# Appendix: hyperparameter calibration
# ---------------------------------------------------------------------------
# Nuclear-norm weights swept for the model-based S-PI (spi, spi_est).
SPI_LAMBDAS = (1, 5, 20, 50, 200, 1000)

# Learning rates swept per policy gradient solver.
PG_LEARNING_RATES = {
    "mb_pg": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1.2e-3),
    "mb_est": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1.2e-3),
    "mf_npg": (3e-4, 1e-3, 3e-3, 1e-2),
}

# Stopping-rule settings for RL runs.
RL_CONVERGENCE_GRID = dict(
    rel_tol=(1e-3, 3e-3, 1e-2, 3e-2),
    patience=(5, 10, 20, 40),
    window=(1, 3, 5, 10),
)


# ---------------------------------------------------------------------------
# Solvers per chapter
# ---------------------------------------------------------------------------
# Sample size solver to compare rollout lengths for (appendix sweep).
CH4_SAMPLE_SOLVERS = ("mf_npg",)

# System ID solver (appendix sweep).
CH4_IDENTIFICATION_SOLVERS = ("mb_est",)

# Every solver whose training budget is an ``(l, m)`` combination
ROLLOUT_SOLVERS = ("mf_npg", "spi_mf")

# Every solver that can learn from an identified model
IDENTIFICATION_SOLVERS = ("mb_est", "ce_dare", "spi_est", "spi_est", "ppo_encoder")

# --- Chapter 4: true model ------------------------------------
CH4_TRUE_SOLVERS = ("baseline", "mb_pg", "spi")

# --- Chapter 4: unknown model ------------------------------
CH4_IDENTIFIED_SOLVERS = (
    "baseline",
    "ce_dare",
    "mb_est",
    "spi_est",
    "mf_npg",
    "spi_mf",
)

# --- Chapter 5: reinforcement learning -------------------------------------
# The inflation sweep.
CH5_SOLVERS = (
    "baseline",
    "ppo_linear",
    "ppo_bottleneck",
    "ppo_encoder",
    "ppo_encoder_scalable",
    "ppo_mlp",
)

# The comparison at CH5_REFERENCE_INFLATION.
CH5_COMPARISON_SOLVERS = (
    "baseline",
    "ppo_linear",
    "ppo_bottleneck",
    "ppo_encoder_scalable",
    "ppo_mlp",
)

# Algorithms that cannot be run at a given inflation size, and why.
UNSUPPORTED_CELLS = {
    ("ppo_encoder", "pixel"): (
        "N4SID cannot identify an 18432-dimensional observation; "
        "ppo_encoder_scalable is replacement"
    ),
}


# Episodes per learning-curve sample for chap 6
PIXEL_PROBE_EPISODES = 100

# Steps collected, and epochs used for pretraining
PRETRAIN_STEPS = 20_000
PRETRAIN_EPOCHS = 20

# Width of the orthonormal projection, as suggested by the paper
BOTTLENECK_DIM = 2

# Used for encoder_scalable and pixel_estimator as the estimated dimension
ENCODER_LATENT_DIM = 2

# Chapter 6: control from a camera observation
CH6_SOLVERS = (
    "oracle_coarse",
    "ppo_pixel_fc",
    "ppo_pixel_ortho",
    "ppo_pixel_estimator",
    "ppo_pixel_posvel",
)


# ---------------------------------------------------------------------------
# Solver hyperparameters
# ---------------------------------------------------------------------------
SOLVER_HYPERPARAMS = {
    "mb_pg": {
        "gradient_lr": 1.2e-3,
    },
    "mf_npg": {
        "gradient_lr": 1e-3,
        "smoothing": 0.5,
    },
    "spi": {
        "lambda_": 50.0,
        "beta": 0.5,
        "epsilon": 1e-8,
        "max_linesearch_iter": 20,
    },
    "spi_mf": {
        "lambda_": 4000,
        "beta": 0.5,
        "epsilon": 1e-8,
        "max_linesearch_iter": None,
        "stepsize": 3.5e-6,
        "smoothing": 0.5,
    },
    # Certainty equivalence. Identification settings only
    "ce_dare": {
        "excitation": 1.0,
        "order_threshold": 1e-3,
        "max_order": 10,
        "horizon": 10,
        "input_rank_tol": 1e-3,
    },
    # S-PI on the identified model
    "spi_est": {
        "excitation": 1.0,
        "order_threshold": 1e-3,
        "max_order": 10,
        "horizon": 10,
        "lambda_": 50.0,
        "beta": 0.5,
        "epsilon": 1e-8,
        "max_linesearch_iter": 10,
    },
    "mb_est": {
        "gradient_lr": 1.2e-3,
        "excitation": 1.0, # std of variation added to input
        "order_threshold": 1e-3,
        "max_order": 10, # maximum order to consider for model identification
        "horizon": 10, # Block Hankel past/future horizon
        "input_rank_tol": 1e-3,
    },
    # Robotic priors for state estimator
    "ppo_encoder_scalable": {
        "alpha": 1.0,
        "beta": 1.0,
        "weights": (1.0, 1.0, 1.0, 2.0),
    },
    # Identification only: ppo_encoder uses N4SID for its ``decode``
    "ppo_encoder": {
        "excitation": 1.0, # std of variation added to input
        "order_threshold": 1e-3,
        "max_order": 10, # maximum order to consider for model identification
        "horizon": 10, # Block Hankel past/future horizon
        "input_rank_tol": 1e-3,
    },
}


# Per inflation size overrides of SOLVER_HYPERPARAMS
# Filled from
# `python -m experiments.figures.appendix_spi_lambda_figures` and
# `python -m experiments.figures.appendix_learning_rate_figures`.
SIZE_HYPERPARAMS = {
    "mb_pg": {
        "base": {"gradient_lr": 1e-05},
        "small": {"gradient_lr": 1e-05},
        "medium": {"gradient_lr": 1e-05},
        "large": {"gradient_lr": 1e-05},
    },
    "mb_est": {
        "base": {"gradient_lr": 0.0012},
        "small": {"gradient_lr": 0.0012},
        "medium": {"gradient_lr": 0.0012},
        "large": {"gradient_lr": 0.0012},
    },
    "mf_npg": {
        "base": {"gradient_lr": 0.01},
        "small": {"gradient_lr": 0.01},
        "medium": {"gradient_lr": 0.01},
        "large": {"gradient_lr": 0.01},
    },
    "spi": {
        "base": {"lambda_": 5},
        "small": {"lambda_": 5},
        "medium": {"lambda_": 5},
        "large": {"lambda_": 5},
    },
    "spi_est": {
        "base": {"lambda_": 20},
        "small": {"lambda_": 20},
        "medium": {"lambda_": 20},
        "large": {"lambda_": 20},
    },
}


# ---------------------------------------------------------------------------
# Output locations, relative to ``code/``
# ---------------------------------------------------------------------------
EXPERIMENT_ROOT = "experiment_results"
GENERATED_ROOT = "../thesis/generated"
FIGURE_OUTPUT = "figures"
TABLE_OUTPUT = "tables"
