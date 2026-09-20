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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Fixed iterations for the LQR solvers.
SOLVER_ITERATIONS = 500

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
    "base": (5, 500),
    "small": (5, 500),
    "medium": (5, 500),
    "large": (5, 2000),
}

SUBOPTIMAL_IDENTIFICATION = {
    "medium": (10, 500),
    "large": (5, 1000),
}

# TODO: RL solvers should be chosen based on the above selection
# Episode length the RL solvers train on.
TRAIN_HORIZON = None

# Environment-step budget for the RL solvers.
RL_TIMESTEPS = None
RL_EVAL_FREQ = None #25_000


# ---------------------------------------------------------------------------
# Noise, applied at test time (chapter 4)
# ---------------------------------------------------------------------------
NOISE_LEVELS = ("none", "weak", "standard", "strong", "divergent")
NOISE_CHANNELS = ("measurement", "control", "parametric", "disturbance")

# Standard deviation per channel and level. ``none`` must stay 0.0.
# Levels based on observation of optimal controller under these conditions.

NOISE_STD = {
    "measurement": {
        "none": 0.0,
        "weak": 0.1,
        "standard": 0.5,
        "strong": 2.0,
        "divergent": 20.0,
    },
    "control": {
        "none": 0.0,
        "weak": 0.1,
        "standard": 0.5,
        "strong": 2.0,
        "divergent": 10.0,
    },
    "parametric": {
        "none": 0.0,
        "weak": 0.0005,
        "standard": 0.001,
        "strong": 0.005,
        "divergent": 0.01,
    },
    "disturbance": {
        "none": 0.0,
        "weak": 0.0005,
        "standard": 0.001,
        "strong": 0.005,
        "divergent": 0.05,
    },
}

# Shape of the two channels that have a mode besides their intensity.
PARAMETRIC_TIME_INVARIANT = True
DISTURBANCE_TYPE = "sinusoidal"  # "gaussian" | "sinusoidal" | "constant"

# The single level the per-channel comparison (table 4, plot 5) is drawn at.
# TODO: set after inspecting the results of `python -m experiments.ch4_noise`
CHOSEN_NOISE_LEVEL = None

# The single channel the per-level comparison (table 3, plot 4) is drawn for.
# TODO: set after inspecting the results of `python -m experiments.ch4_noise`
CHOSEN_NOISE_CHANNEL = None


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
CH4_REFERENCE_INFLATION = "medium"

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
    "mf_npg",
    "ppo_linear",
    "ppo_bottleneck",
    "ppo_encoder_scalable",
    "ppo_mlp",
)

# Cells without a budget of their own
BORROWED_BUDGETS = {
    ("mf_npg", "pixel"): "large",
}

# Algorithms that cannot be run at a given inflation size, and why.
UNSUPPORTED_CELLS = {
    ("ppo_encoder", "pixel"): (
        "N4SID cannot identify an 18432-dimensional observation; "
        "ppo_encoder_scalable is replacement"
    ),
}


# Episodes per learning-curve sample for chap 6
PIXEL_PROBE_EPISODES = 100

# Frames collected, and epochs used for chap 6 pretraining.
PIXEL_PRETRAIN_STEPS = 20_000
PIXEL_PRETRAIN_EPOCHS = 20

# Chapter 6: control from a camera observation
CH6_SOLVERS = (
    "oracle_coarse",
    "ppo_pixel_fc",
    "ppo_pixel_ortho",
    "ppo_pixel_estimator",
    "ppo_pixel_posvel",
    "ce_dare_pixel",
    "spi_est_pixel",
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
        "max_linesearch_iter": 10,
    },
    "spi_mf": {
        "lambda_": 4e4,
        "beta": 0.5,
        "epsilon": 1e-8,
        "max_linesearch_iter": 10,
        "stepsize": 3.5e-7,
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
    # Identification only: ppo_encoder uses N4SID for its ``decode``
    "ppo_encoder": {
        "excitation": 1.0, # std of variation added to input
        "order_threshold": 1e-3,
        "max_order": 10, # maximum order to consider for model identification
        "horizon": 10, # Block Hankel past/future horizon
        "input_rank_tol": 1e-3,
    },
}


# ---------------------------------------------------------------------------
# Output locations, relative to ``code/``
# ---------------------------------------------------------------------------
EXPERIMENT_ROOT = "experiment_results"
GENERATED_ROOT = "../thesis/generated"
FIGURE_OUTPUT = "figures"
TABLE_OUTPUT = "tables"
