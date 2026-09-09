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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Fixed iterations for the LQR solvers.
SOLVER_ITERATIONS = 500

# Size of the full-rank component of the initial gain on an inflated system.
K0_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Sample sizes (chapter 4)
# ---------------------------------------------------------------------------
# l = rollout lengths, m = trajectory counts.
ROLLOUT_LENGTHS = (200, 500, 800)
TRAJECTORY_COUNTS = (100, 200, 500)
SAMPLE_GRID = tuple((l, m) for l in ROLLOUT_LENGTHS for m in TRAJECTORY_COUNTS)

# Reduced rollout lengths for the larger inflations
SAMPLE_GRID_FULL_SIZES = ("base", "small")
SAMPLE_GRID_REDUCED = tuple((l, m) for l in (200,) for m in TRAJECTORY_COUNTS)


def sample_grid_for(size: str) -> tuple[tuple[int, int], ...]:
    """The (l, m) cells swept at one inflation size."""
    return SAMPLE_GRID if size in SAMPLE_GRID_FULL_SIZES else SAMPLE_GRID_REDUCED

# Per inflation size sample sizes for model-free approach 

# TODO: fill from `python -m experiments.ch4_sample_size`; values are (l, m).
CHOSEN_SAMPLE_CELL = {
    "base": None,
    "small": None,
    "medium": None,
    "large": None,
}

# Length of excitation rollouts
IDENTIFICATION_LENGTHS = (500, 1000, 2000, 4000, 8000)

# Past/future horizon of the block Hankel matrices for N4SID
IDENTIFICATION_HORIZONS = (5, 10, 20)

# TODO: fill from `python -m experiments.ch4_identification_length`;
# values are (trajectory_length, hankel_horizon).
CHOSEN_IDENTIFICATION = {
    "base": None,
    "small": None,
    "medium": None,
    "large": None,
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
# TODO: fill in the magnitude of the levels
NOISE_STD = {
    "measurement": {
        "none": 0.0,
        "weak": None,
        "standard": None,
        "strong": None,
        "divergent": None,
    },
    "control": {
        "none": 0.0,
        "weak": None,
        "standard": None,
        "strong": None,
        "divergent": None,
    },
    "parametric": {
        "none": 0.0,
        "weak": None,
        "standard": None,
        "strong": None,
        "divergent": None,
    },
    "disturbance": {
        "none": 0.0,
        "weak": None,
        "standard": None,
        "strong": None,
        "divergent": None,
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
}

# The size comparisons of chapter 5 will be run at
REFERENCE_INFLATION = "medium"


# ---------------------------------------------------------------------------
# Solvers per chapter
# ---------------------------------------------------------------------------
# Sample size solver to compare rollout lengths for
CH4_SAMPLE_SOLVERS = ("mf_npg",)

# System ID solver
CH4_IDENTIFICATION_SOLVERS = ("mb_est",)

# What we evaluate on in the end
CH4_SOLVERS = ("baseline", "mb_pg", "mb_est", "mf_npg")

# Compare the effect of different inflation sizes on api
CH5_INFLATION_SOLVERS = ("spi",)

# Do the same for RL approach with only inflation sizes
CH5_RL_SOLVERS = ("ppo_encoder",)

# Final evaluation in chapter 5
CH5_SOLVERS = (
    "baseline",
    "mb_est",
    "spi",
    "ppo_linear",
    "ppo_mlp",
    "ppo_encoder",
)

# Goal for chap 6
CH6_SOLVERS = ("baseline", "ppo_encoder", "pve")


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
    "mb_est": {
        "gradient_lr": 1.2e-3,
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
FIGURE_OUTPUT = "../thesis/generated/figures"
TABLE_OUTPUT = "../thesis/generated/tables"
