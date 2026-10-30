# Low-Dimensional Control under Embodiment Constraints: From LQR to Reinforcement Learning

Code and LaTeX source of the Bachelor thesis by Alexander Michael Kranz, written in 2026 at the Hamburg
University of Technology (TUHH). The compiled thesis is [`thesis/thesis.pdf`](thesis/thesis.pdf).

The thesis asks how enforcing low-dimensional structure in a controller affects its performance and its robustness
to noise when the embodiment offers far more sensors and actuators than the dynamics need. The plant is a double
integrator throughout. Its state and input are *artificially inflated* through fixed lifts into a much larger
observation and action space, up to a 96×96 camera image driving a ring of 256 thrusters. Three families of
controllers are compared against the analytic LQR solution:

1. **LQR solvers** (thesis Ch. 3): policy gradient and Structured Policy Iteration (S-PI, a nuclear-norm penalty
   that recovers the low-rank gain), with the true model, an identified model (N4SID), or no model at all.
2. **Reinforcement learning** (thesis Ch. 4): PPO with linear, MLP, bottleneck and encoder policies across
   inflation sizes.
3. **Perception-based control** (thesis Ch. 5): PPO on the pixel image of the double integrator.

Each is evaluated under four noise channels (measurement, control, parametric, disturbance) at calibrated intensities.

All code is either adapted from the sources cited in the code and in the thesis, or the author's own work.

> **Chapter numbers.** The code was written before the thesis chapters were numbered, so file names are one ahead
> of the thesis: **code `ch4_*` / `ch5_*` / `ch6_*` = thesis Chapters 3 / 4 / 5**. The code's `appendix_*`
> experiments are thesis Appendices A–C.

---

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Quick start](#quick-start)
- [How an experiment works](#how-an-experiment-works)
- [Reproducing the thesis](#reproducing-the-thesis)
- [Where each result appears in the thesis](#where-each-result-appears-in-the-thesis)
- [Conventions](#conventions)
- [References](#references)

---

## Repository layout

```
.
├── code/                            all Python code; every command runs from here
│   ├── settings.py                  every experiment parameter, incl. the values chosen by the sweeps
│   ├── config.py                    the double integrator, dt, initial-state box, camera constants
│   ├── seeding.py                   seed sequences derived from the seeds in settings.py
│   ├── basetypes.py                 small shared dataclasses
│   ├── lqr_base/                    the plant
│   ├── lqr_solvers/                 the LQR solvers and the system identification
│   ├── reinforcement_learning/      Gymnasium environments, PPO solvers and policy architectures
│   ├── evaluation/                  metrics, probes, run records
│   ├── experiments/                 one runnable script per experiment, plus the shared pipeline
│   │   └── figures/                 one figure script per experiment, writing into thesis/generated/
│   ├── cluster/                     SLURM submission, result sync, and the full workflow (WORKFLOW.md)
│   ├── experiment_results/          stored runs (git-ignored, created on demand)
│   └── results/                     output of the early interactive helpers (git-ignored)
├── thesis/                          LaTeX source and the compiled thesis.pdf
│   ├── generated/                   figures (.pgf/.pdf) and tables (.tex) written by the figure scripts
│   └── figures/                     hand-drawn TikZ figures
├── project_description/             the original project description
└── requirements.txt
```

### The code in more detail

| Path | Contents |
| --- | --- |
| [`code/settings.py`](code/settings.py) | The single place for parameters: seeds, budgets, inflation sizes, noise levels, solver hyperparameters. Values chosen by an experiment carry a *Filled from* comment naming it. |
| [`code/config.py`](code/config.py) | The double integrator `A`, `B`, `dt = 0.01`, the initial-state box, and the camera environment's constants (frame size, action repeat, thruster ring). |
| [`code/lqr_base/simple_lqr.py`](code/lqr_base/simple_lqr.py) | `LQRDiscrete`, the (optionally inflated) plant; `get_optimal_K_and_cost_discrete`, the exact Riccati solution every result is measured against; `true_cost`; `simulate`. |
| [`code/lqr_base/noise.py`](code/lqr_base/noise.py) | The four noise channels. |
| [`code/lqr_base/thruster_ring.py`](code/lqr_base/thruster_ring.py) | The thruster-ring actuation of the camera environment and its coarse-rate LQR oracle. |
| [`code/lqr_solvers/mb_policy_gradient.py`](code/lqr_solvers/mb_policy_gradient.py) | Model-based policy gradient (exact gradient). |
| [`code/lqr_solvers/mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) | Model-free (natural) policy gradient with a two-point estimator. |
| [`code/lqr_solvers/structured_policy_iteration.py`](code/lqr_solvers/structured_policy_iteration.py) | S-PI, model-based and model-free. |
| [`code/lqr_solvers/system_identification.py`](code/lqr_solvers/system_identification.py) | N4SID identification (via SIPPY) and the surrogate plant the model-based solvers then run on. |
| [`code/reinforcement_learning/lqr_env.py`](code/reinforcement_learning/lqr_env.py) | The plant as a Gymnasium environment. |
| [`code/reinforcement_learning/pixel_env.py`](code/reinforcement_learning/pixel_env.py), [`pixel_render.py`](code/reinforcement_learning/pixel_render.py) | The camera environment and its rasterizer. |
| [`code/reinforcement_learning/solvers.py`](code/reinforcement_learning/solvers.py), [`features.py`](code/reinforcement_learning/features.py) | The PPO presets and the policy architectures as Stable-Baselines3 feature extractors. |
| [`code/evaluation/`](code/evaluation) | Cost metrics, per-iteration probes, learning-curve analysis, the latent probe (how well a policy recovers the true state), and the JSON + NPZ run records. |
| [`code/experiments/`](code/experiments) | The experiment scripts (`ch*_*.py`, `appendix_*.py`) and the pipeline they share: `problem.py` (what a run is), `registry.py` (all solvers), `runner.py` (train a grid), `evaluate.py` (the test protocol), `store.py` (storage and sharding), `policies.py` (reloading a stored policy for the noise experiments). |
| [`code/experiments/figures/`](code/experiments/figures) | One `<experiment>_figures.py` per experiment, plus shared aggregation, panels, tables and the LaTeX style. |
| [`code/cluster/`](code/cluster) | [`WORKFLOW.md`](code/cluster/WORKFLOW.md) (the complete reproduction guide), `submit.sh`, `sync.sh`, and maintenance scripts for older records. |
| [`code/visualization.py`](code/visualization.py), [`code/evaluation/plots.py`](code/evaluation/plots.py) | Early interactive plotting helpers that write to `code/results/`. Not used for any figure in the thesis. |

---

## Setup

**Requirements:** Python ≥ 3.12 (the results were produced with 3.12.13) and, for the figure scripts and the thesis,
a LaTeX installation with `pdflatex` and `latexmk` (TeX Live 2026 was used). The figure scripts render through
Matplotlib's `pgf` backend, so they fail without `pdflatex`. Running experiments does not need LaTeX.

```bash
# in the repository root
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# optional, on a machine without a GPU: the CPU build of torch avoids a ~2 GB CUDA download
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) lists the lower bounds the code needs and, in comments, the exact versions the
thesis results were produced with. All packages except the commented-out `pygame` are needed for a full
reproduction: `torch` is imported by every experiment, `stable-baselines3` and `gymnasium` by the RL and camera
solvers, and `sippy_unipi` by the identified-model solvers.

Check the installation:

```bash
cd code
python -c "import numpy, scipy, matplotlib, torch, gymnasium, stable_baselines3, sippy_unipi, tqdm; print('ok')"
```

For running on the TUHH HPC cluster (SSH, Python 3.12 via conda, SLURM partitions), see §0 of
[`code/cluster/WORKFLOW.md`](code/cluster/WORKFLOW.md).

---

## Quick start

All commands run **from `code/`**, as modules (`python -m ...`): the code imports absolutely from `code/`
(`import settings`, `from lqr_base...`), and the result and output paths in `settings.py` are relative to it.

The cheapest complete experiment is the true-model comparison of thesis §3.1 (about 12 minutes of CPU time in total):

```bash
cd code
python -m experiments.ch4_true_model                    # trains all runs into experiment_results/ch4_true_model/
python -m experiments.figures.ch4_true_model_figures    # writes thesis/generated/ch4_true_model/
cd ../thesis && latexmk -pdf thesis.tex                 # rebuilds the thesis with the new figures
```

`settings.py` ships with every value the thesis used, so any experiment can be run directly like this, without
first rerunning the sweeps that chose those values. Note that the figure scripts **overwrite** the committed
figures and tables in `thesis/generated/`.

Other experiments that are feasible on a laptop: `appendix_spi_lambda` (≈ 1 CPU-hour),
`appendix_identification_length` (≈ 3 CPU-hours) and the noise experiments `ch4_true_model_noise` / `ch4_noise`
(< 1 hour, but they need their source experiment first). Everything involving model-free solvers at large inflation,
RL or the camera takes hundreds to thousands of CPU-hours; see the [cost overview](#cost) below.

---

## How an experiment works

Every experiment script in `code/experiments/` builds a grid of runs (solver × inflation size × seed, or a
hyperparameter sweep) and hands it to `experiments.runner.run_grid`. For each run it

1. builds the plant and solver from `settings.py`,
2. trains the solver and records its learning curve,
3. evaluates the final policy on the fixed test protocol (`TEST_EPISODES = 100` episodes of `TEST_HORIZON = 2000`
   steps),
4. stores the run as `experiment_results/<experiment>/<run>/record.json` + `arrays.npz` (plus `policy.zip` for PPO).

Some behaviour worth knowing before running anything:

- **Finished runs are skipped**, so an interrupted experiment is resumed by starting it again. A run that was cut
  off restarts from zero.
- **The settings are guarded.** Each experiment writes the settings it used to a `manifest` in its result folder.
  Changing one of them afterwards raises `SettingsChangedError` instead of silently mixing runs from different
  settings. To start over, delete `experiment_results/<experiment>/`.
- **Sharding** is controlled by the SLURM variables `SLURM_ARRAY_TASK_COUNT` / `SLURM_ARRAY_TASK_ID`. Without them
  (i.e. locally) a script runs its whole grid.
- **Missing values stop the run.** If a value an experiment depends on is set to `None` in `settings.py`, it stops
  with a message naming the experiment that chooses it.
- On macOS a running experiment keeps the machine awake (`caffeinate`).

The noise experiments (`*_noise`) train nothing: they reload the stored policies of one main experiment and score
them under every noise channel and level, at test time only.

---

## Reproducing the thesis

The experiments depend on each other in two ways: some choose values in `settings.py` that later ones read
(budgets, noise intensities, hyperparameters), and the noise experiments score the policies stored by the main
experiments. This gives four waves; everything within a wave is independent.

```text
Wave A   budgets and noise intensities
         appendix_sample_size, appendix_identification_length,
         appendix_calibrate_noise, appendix_calibrate_pixel_noise
            │  → CHOSEN_SAMPLE_CELL, CHOSEN_IDENTIFICATION, NOISE_STD, CH6_NOISE_STD
            ▼
Wave B   hyperparameter sweeps
         appendix_learning_rate, appendix_spi_lambda, appendix_spi_mf, appendix_rl_learning_rate
            │  → SIZE_HYPERPARAMS
            ▼
Wave C   main experiments
         ch4_true_model, ch4_identified_model, ch5_inflation_sizes,
         ch6_solver_comparison, ch6_perception, appendix_rl_convergence
            │  → stored policies
            ▼
Wave D   noise experiments
         ch4_true_model_noise, ch4_noise, ch5_noise, ch6_noise
```

For each experiment:

```bash
python -m experiments.<experiment>                          # train (locally, or via cluster/submit.sh)
python -m experiments.figures.<figure script>               # figures + tables, see the table below
```

After waves A and B, the figure script of each sweep **prints the value it chooses**. If it matches the one in
`settings.py`, go on; if not, paste it in before starting the next wave. The camera's learning rates and
measurement-noise levels are set by hand, not chosen by a sweep; [`WORKFLOW.md`](code/cluster/WORKFLOW.md) §3.2
explains how. After `ch4_noise`, `CHOSEN_NOISE_LEVEL` and `CHOSEN_NOISE_CHANNEL` pick which level and channel the
Ch. 3 comparison figures show; they change figures only, not runs.

The full guide, with the exact `submit.sh` commands, shard counts, time limits and how to chain the waves on the
cluster, is [`code/cluster/WORKFLOW.md`](code/cluster/WORKFLOW.md) §1–§3.

<a id="cost"></a>
### Cost

Approximate CPU time of each experiment on one core (from WORKFLOW.md §2; the RL budgets are fixed, so these are
predictable):

| Wave | Experiment | CPU time |
| --- | --- | --- |
| A | `appendix_sample_size` | 286 h |
| A | `appendix_identification_length` | 3 h |
| A | `appendix_calibrate_noise` | ≤ 6 h per channel |
| A | `appendix_calibrate_pixel_noise` | ≈ 20 h |
| B | `appendix_learning_rate` | 309 h |
| B | `appendix_spi_lambda` | 1 h |
| B | `appendix_spi_mf` | 141 h |
| B | `appendix_rl_learning_rate` | 2030 h |
| C | `ch4_true_model` | 0.2 h |
| C | `ch4_identified_model` | 108 h |
| C | `ch5_inflation_sizes` | 1210 h |
| C | `ch6_solver_comparison` | 1310 h |
| C | `ch6_perception` | 1960 h (single runs up to ≈ 4 days) |
| C | `appendix_rl_convergence` | 370 h |
| D | `ch4_true_model_noise`, `ch4_noise`, `ch5_noise` | < 1 h each |
| D | `ch6_noise` | ≈ 17 h |

In total about 7 800 CPU-hours. On the TUHH cluster, with every run as its own array task, the full progression from
an empty `experiment_results/` took about 1.5 weeks plus queue time.

The stored runs (≈ 20 GB) are not part of this repository. The figures and tables they produced are, in
[`thesis/generated/`](thesis/generated).

---

## Where each result appears in the thesis

| Thesis | Experiment (`python -m experiments.<…>`) | Figure script (`python -m experiments.figures.<…>`) |
| --- | --- | --- |
| §3.1 A Known Model | `ch4_true_model` | `ch4_true_model_figures` |
| §3.1, under noise | `ch4_true_model_noise` | `ch4_true_model_noise_figures` |
| §3.2 Unknown Model across Dimensions | `ch4_identified_model` | `ch4_identified_model_figures` |
| §3.2, under noise | `ch4_noise` | `ch4_noise_figures` |
| §4.2 Comparison across Dimensions | `ch5_inflation_sizes` | `ch5_inflation_figures` |
| §4.3 Noise on Large Inflation | `ch5_noise` | `ch5_noise_figures` |
| §5.1 Artificial Inflation to Pixel Size | `ch6_solver_comparison` | `ch6_solver_comparison_figures` |
| §5.2 Effect of Visual Inputs on the Solvers | `ch6_perception` | `ch6_perception_figures` |
| §5.3 Noisy Perception | `ch6_noise` (intensities from `appendix_calibrate_pixel_noise`) | `ch6_noise_figures` |
| App. A, noise intensities | `appendix_calibrate_noise`, `appendix_calibrate_pixel_noise` | `appendix_calibrate_noise_figures`, `appendix_calibrate_pixel_noise_figures` |
| App. C, learning rate of TrP, IdP, MfN | `appendix_learning_rate` | `appendix_learning_rate_figures` |
| App. C, nuclear-norm weight of S-PI | `appendix_spi_lambda` | `appendix_spi_lambda_figures` |
| App. C, N4SID identification length | `appendix_identification_length` | `appendix_identification_figures` |
| App. C, sample size | `appendix_sample_size` | `appendix_sample_size_figures` |
| App. C, step size and weight of MfS | `appendix_spi_mf` | `appendix_spi_mf_figures` |
| App. C, learning rates of the RL solvers | `appendix_rl_learning_rate` | `appendix_rl_learning_rate_figures` |
| App. C, stopping rule of the RL solvers | `appendix_rl_convergence` | `appendix_rl_convergence_figures` |

App. B collects additional plots from `ch4_true_model`, `ch4_noise`, `ch6_perception` and `ch6_noise`, made by the
same figure scripts.

### Solver names

The code names solvers by their registry key (`experiments/registry.py`); the thesis uses short labels
(`experiments/figures/aggregate.py`):

| Thesis | Code | Solver |
| --- | --- | --- |
| Baseline | `baseline` | exact LQR solution (discrete Riccati equation) |
| TrP / TrS | `mb_pg` / `spi` | policy gradient / S-PI on the true model |
| IdP / IdS / IdD | `mb_est` / `spi_est` / `ce_dare` | policy gradient / S-PI / certainty-equivalent Riccati on the N4SID-identified model |
| MfN / MfS | `mf_npg` / `spi_mf` | model-free natural policy gradient / model-free S-PI |
| Lin, MLP | `ppo_linear`, `ppo_mlp` | PPO with a linear and an MLP policy |
| OrB | `ppo_bottleneck`, `ppo_pixel_ortho` | PPO with an orthonormal bottleneck |
| StE | `ppo_encoder` | PPO behind a state encoder taken from the N4SID-identified output matrix |
| ScE | `ppo_encoder_scalable`, `ppo_pixel_estimator` | PPO behind a pretrained, frozen scalable state encoder |
| FcE | `ppo_fc_extractor`, `ppo_pixel_fc` | PPO with a fully connected feature extractor |
| PVE | `ppo_pixel_posvel` | PPO with a position–velocity encoder (camera only) |
| Oracle | `oracle_coarse` | LQR on the true state at the camera's control rate (camera only) |

---

## Conventions

A few things that are assumed everywhere and are easy to get wrong when reading the code.

- **Discrete time.** Continuous `A`, `B` and `dt` go in; `LQRDiscrete` discretizes with a zero-order hold.
- **Feedback is `v = -K y`.** The source papers write `u = +K x`; the sign is absorbed once, in the exact solution
  and the closed loop, so no solver carries it.
- **Inflation is two fixed lifts.** With `U_1`, `U_2` in its settings, the plant observes `y = U_1 x` and actuates
  `u = U_2 v`:

  ```
  x  --[ U_1 ]-->  y  --[ policy ]-->  v  --[ U_2 ]-->  u
  (n)             (p)                 (q)              (m)
  ```

  The learned gain `K` has shape `(q, p)`, while the gain acting on the plant is `U_2 K U_1`, of rank at most
  `min(n, m)`. The cost is charged in the controller's space (`Q`: p×p, `R`: q×q). The lifts are orthonormal
  (built by QR); plain random lifts made the solvers diverge. The sizes `(p, q)` are `INFLATION_SIZES` in
  `settings.py`: base (2, 1), small (10, 5), medium (50, 25), large (200, 100), and pixel (18432, 256), i.e. two
  stacked 96×96 frames and 256 thrusters.
- **Noise acts wherever the plant is stepped** (in training rollouts, the Gymnasium environment and the test
  protocol) but never inside the model-based solvers, which work from `A` and `B` directly. Measurement noise is
  additive on `y`, control noise multiplicative on `v`, parametric noise perturbs `A` and `B`, and the disturbance
  (sinusoidal in the thesis) acts on the successor state.
- **One step counter.** Every transition goes through `LQRBase.apply_control`, which the solvers and the Gymnasium
  environments share, so model-free solvers and RL are measured on the same environment-step budget.
- **Seeds.** The main experiments train every configuration on the ten `SEEDS` in `settings.py`
  (`appendix_rl_convergence` on the first three), and the results pool the test episodes of all seeds.

---

## References

The main methods implemented here; the thesis bibliography has the complete list.

| Source | Used in |
| --- | --- |
| Park, Rossi, Wen, Wu, Zhao: [*Structured Policy Iteration for Linear Quadratic Regulator*](https://arxiv.org/abs/2007.06202), ICML 2020 | [`structured_policy_iteration.py`](code/lqr_solvers/structured_policy_iteration.py) |
| Fazel, Ge, Kakade, Mesbahi: [*Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator*](https://arxiv.org/abs/1801.05039), ICML 2018 | [`mb_policy_gradient.py`](code/lqr_solvers/mb_policy_gradient.py), [`mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) |
| Van Overschee, De Moor: *N4SID: Subspace Algorithms for the Identification of Combined Deterministic-Stochastic Systems*, Automatica 1994; implementation from [SIPPY](https://github.com/CPCLAB-UNIPI/SIPPY) | [`system_identification.py`](code/lqr_solvers/system_identification.py) |
| Schulman et al.: [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017; implementation from [Stable-Baselines3](https://stable-baselines3.readthedocs.io) | [`reinforcement_learning/`](code/reinforcement_learning) |
| [Gymnasium](https://gymnasium.farama.org) | [`lqr_env.py`](code/reinforcement_learning/lqr_env.py), [`pixel_env.py`](code/reinforcement_learning/pixel_env.py) |

---

Parts of the code and of this README were written with the help of Claude Code; the declaration at the end of the
thesis describes how generative AI was used.
