# Low-dimensional Control under Embodiment Constraints: From LQR to Reinforcement Learning

Code for the Bachelor Thesis written by Alexander Kranz in 2026 at TUHH.
All the code is either adapted from the indicated sources, or the own work of the author. This README has been restructured with the help of Claude Code.

The repository studies how a controller behaves when its **embodiment** does not match its
underlying model: when it observes a small system through many redundant sensors and drives it through
many redundant actuators, and when both are noisy. The testbed is a double integrator
throughout. Around it, three families of methods are implemented and compared against the
analytic LQR solution:

1. **Policy gradient** — model-based and model-free, as the unstructured reference.
2. **Structured Policy Iteration** — a nuclear-norm penalty that recovers the low-rank
   controller the inflated problem actually admits.
3. **Reinforcement learning** — the same plant as a Gymnasium environment (early stage).

> **Status:** work in progress. Open items are tracked in
> [`code/notes/PROCESS.md`](code/notes/PROCESS.md) and summarised under
> [Status and open items](#status-and-open-items).

---

## Repository layout

```
.
├── code/
│   ├── basetypes.py                  shared dataclasses + the global SEED
│   ├── visualization.py              all plotting
│   ├── lqr_base/                     the systems being controlled
│   │   ├── simple_lqr.py             discrete LQR, exact solution, simulator
│   │   ├── inflated_lqr.py           LQR seen through lifted observation/action spaces
│   │   └── noise.py                  the four noise channels
│   ├── lqr_solvers/                  methods for finding a policy
│   │   ├── mb_policy_gradient.py     model-based gradient descent
│   │   ├── mf_policy_gradient.py     model-free (natural) policy gradient
│   │   └── structured_policy_iteration.py   S-PI, model-based and model-free
│   ├── evaluation/                   runnable experiments
│   │   ├── policy_gradient_evaluation.py    solver comparison
│   │   ├── noise_evaluation.py       one noise channel at a time
│   │   ├── inflate_evaluation.py     full-rank vs. low-rank under inflation
│   │   └── spi_evaluation.py         does S-PI actually reduce the rank?
│   ├── reinforcement_learning/       the RL
│   │   ├── double_integrator_environment.py   Gymnasium env
│   │   └── inflated_rl.py            PPO prototype
│   └── notes/                        working notes, not documentation
├── thesis/                           LaTeX source of the thesis itself
└── requirements.txt
```

| Path | Purpose |
| --- | --- |
| [`code/basetypes.py`](code/basetypes.py) | `StateSpaceModel`, `StepResult`, `SimulationResult`, and the `SEED` constant. No logic. |
| [`code/visualization.py`](code/visualization.py) | Functions for visualizing outputs of experiments |
| [`code/lqr_base/simple_lqr.py`](code/lqr_base/simple_lqr.py) | `LQRDiscrete`, the exact solution `get_optimal_K_and_cost_discrete`, `true_cost`, `simulate`. |
| [`code/lqr_base/inflated_lqr.py`](code/lqr_base/inflated_lqr.py) | `LQRInflated` - the model behind two lift matrices for increasing dimensions. |
| [`code/lqr_base/noise.py`](code/lqr_base/noise.py) | Parametric, control, measurement and disturbance noise. |
| [`code/lqr_solvers/mb_policy_gradient.py`](code/lqr_solvers/mb_policy_gradient.py) | `gradient_descent_discrete` - exact gradient. |
| [`code/lqr_solvers/mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) | `model_free_policy_gradient`, `model_free_natural_policy_gradient`, and the shared `estimate_gradient`. |
| [`code/lqr_solvers/structured_policy_iteration.py`](code/lqr_solvers/structured_policy_iteration.py) | `structured_policy_iteration` and `structured_policy_iteration_mf`. |
| [`code/evaluation/`](code/evaluation) | Different runnable experiments. |
| [`code/reinforcement_learning/`](code/reinforcement_learning) | Gymnasium environment and a PPO training script. |
| [`code/notes/PROCESS.md`](code/notes/PROCESS.md) | Journal: what broke, what fixed it, dated. |
| [`code/notes/Plan.md`](code/notes/Plan.md) | Intended order of work and open questions, will be deleted later. |
| [`code/notes/RL_Docs.md`](code/notes/RL_Docs.md) | Links to the Stable-Baselines3 and Gymnasium docs for quick access. |
| [`thesis/`](thesis) | LaTeX source |

---

## Getting started

### Prerequisites

Python 3.12 or newer (tested with 3.12.13). Everything else is in
[`requirements.txt`](requirements.txt):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The RL dependencies (`gymnasium`, `stable-baselines3`, `pygame`) are only needed by
[`code/reinforcement_learning/`](code/reinforcement_learning). The LQR half runs on
`numpy`, `scipy` and `matplotlib` alone.

### Running anything

Scripts import absolutely from `code/` (`from basetypes import ...`), and `code/` itself is
not a package while its subfolders are. So run them **as modules, from `code/`**:

```bash
cd code
python -m evaluation.inflate_evaluation
```

### A note on figure output

Every `run_*` method takes a `visualize` flag:

- `visualize=True` (default) opens the figure in a window.
- `visualize=False` writes it as `Run_<label>_<timestamp>.png`.

Saved figures go to `../../Visualizations/`, which from `code/` resolves to a folder
**outside this repository**, next to the repo root. The directory is not created
automatically.

---

## Conventions

Five things are assumed everywhere and are easy to get wrong when reading the code.

**1. Feedback is `u = -K x`.** Both source papers write `u = +K x`. The sign is absorbed
once, in `get_optimal_K_and_cost_discrete` and in the closed loop `A - B K`, so no solver
carries it. Comments marked `SHOW:` flag the places where the papers' minus was dropped.

**2. Everything is discrete time.** You pass *continuous* `A` and `B` plus a `dt`; each LQR
constructor discretizes with a zero-order hold via `continuous_to_discrete`. Discrete was
chosen because it is what an actual embodied controller runs in.

**3. Inflation means two lift matrices.** `LQRInflated` observes `y = U_1 x` and actuates
`u = U_2 v`:

```
x  --[ U_1 ]-->  y  --[ policy -K ]-->  v  --[ U_2 ]-->  u
(n)             (p)                    (q)              (m)
```

The learned `K` has shape `(q, p)`, but the gain acting on the plant is the **effective
gain** `U_2 @ K @ U_1`, available as `lqr.effective_gain(K)`. This distinction decides which
gain to pass where:

| Function | Takes |
| --- | --- |
| `true_cost(lqr, K)` | the **effective** gain - it evaluates the system |
| `simulate(lqr, K, n)` | the **raw** gain - it drives the controller |
| the solvers | the **raw** gain — they optimize in the controller's space |

Since `U_1` and `U_2` have rank at most `n` and `m`, the optimal inflated `K` is low-rank by
construction — which is exactly what makes S-PI the right tool. The experiments build the
lifts as *orthonormal* (QR) matrices; experiments with
plain random lifts showed that the solvers diverge.

**4. Seeding is used for the initial condition only.** `SEED = #` in [`basetypes.py`](code/basetypes.py) is applied in
`LQRBase.reset()` before the *first* initial-state sample only. With
`resample_initial_state=False` that one sample is reused, so the run is
reproducible; with `True`, later samples, for example during trajectory runs, are resampled, and thus vary overall.

**5. Noise is a runtime flag, not configuration.** Configuring an `LQRNoiseSettings` does not
switch noise on. `reset()` clears the flag and `apply_noise()` sets it, which means noise is
active during `simulate()` and inside `estimate_gradient`'s rollouts, and nowhere else. The
model-based solvers never see it. (This answers the open question at the bottom of
[`Plan.md`](code/notes/Plan.md).) -> TODO: Fix this 

---

## Modules

### `lqr_base` — the systems

[`simple_lqr.py`](code/lqr_base/simple_lqr.py) holds `LQRBase` and the concrete
`LQRDiscrete`, which own the system, the cost weights and the current state, and expose one
operation to a solver: `next_state(K)`. Alongside them are the three free functions the
experiments lean on:

| Function | Role |
| --- | --- |
| `get_optimal_K_and_cost_discrete(lqr)` | Exact solution via the discrete Riccati equation. The baseline everything is measured against. |
| `true_cost(lqr, K)` | Exact infinite-horizon cost of a gain, or `inf` if the closed loop is unstable. Used for every cost curve. |
| `simulate(lqr, K, n_steps)` | Rolls the closed loop out **with noise armed** and records the trajectory. |

[`inflated_lqr.py`](code/lqr_base/inflated_lqr.py) subclasses `LQRDiscrete` and overrides
only what the lifts change: `effective_gain`, `project_gradient` and `next_state`.

[`noise.py`](code/lqr_base/noise.py) holds the four channels. They are plain classes that
know how to sample themselves, bundled into an `LQRNoiseSettings`:

| Channel | Where it acts | Character |
| --- | --- | --- |
| `MeasurementNoiseSettings` | on the observation, before the gain | additive; the cost is still charged on the true state |
| `ControlNoiseSettings` | on the commanded input | **multiplicative** — the error scales with the control and vanishes as the system settles |
| `ParametricNoiseSettings` | on `A` and `B` | a misidentificaiton of the model |
| `DisturbanceNoiseSettings` | on the successor state | `"gaussian"`, `"sinusoidal"` or `"constant"` external force |

For the inflated system, measurement and control noise act in the *lifted* spaces, while
parametric and disturbance noise act on the true plant — sensor and actuator noise are
properties of the embodiment, not of the physics.

### `lqr_solvers` — finding a policy

All solvers return `(ks, costs)`: the full sequence of iterates, so convergence can be
plotted, and a per-iteration cost. Hyperparameters come in as a settings dataclass defined
in the same file.

| Entry point | File | Model-Based? | Notes |
| --- | --- | --- | --- |
| `gradient_descent_discrete` | [`mb_policy_gradient.py`](code/lqr_solvers/mb_policy_gradient.py) | yes | Exact gradient. The clean reference. Does not go through `effective_gain`, so it is for the plain system only. |
| `model_free_policy_gradient` | [`mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) | no | Two-point estimator over `m` rollout pairs. |
| `model_free_natural_policy_gradient` | [`mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) | no | Preconditioned by the inverse observation covariance. Much better conditioned under inflation. |
| `structured_policy_iteration` | [`structured_policy_iteration.py`](code/lqr_solvers/structured_policy_iteration.py) | yes | Proximal gradient with a nuclear-norm penalty. TODO: **Returns an empty cost list** and does not yet work under inflation. |
| `structured_policy_iteration_mf` | [`structured_policy_iteration.py`](code/lqr_solvers/structured_policy_iteration.py) | no | No model, tolerates noise, produces a low-rank gain. |

Three implementation details carry most of the model-free quality, all recorded as hard-won
in [`PROCESS.md`](code/notes/PROCESS.md):

- **Two-point estimation** — evaluating both `K + U` and `K - U` cancels the zeroth-order
  term and cuts the variance sharply.
- **Common random numbers** — the RNG state is saved and restored between the two rollouts
  of a pair, so their difference is attributable to the perturbation and not to the noise.
- **Truncated pseudoinverse** — `pinv(sigma, rcond=1e-6)` in the natural gradient. Under
  inflation `sigma` is rank-deficient by construction, and a plain inverse amplifies
  numerical noise into an enormous step.

For S-PI, `lambda_` and the step size are tightly coupled and cannot be tuned independently:
the singular-value threshold is their product. `PROCESS.md` records the empirically stable
pairs, and they are reproduced in the `StructuredPolicySettings` docstring.

### `evaluation` — the experiments

Each script is runnable and defines its own problem in a `__main__` block. Three of the four
are built around an experiment class holding everything that stays fixed, so that the
`run_*` methods differ only in the one variable under study.

### `reinforcement_learning` — the RL half

[`double_integrator_environment.py`](code/reinforcement_learning/double_integrator_environment.py)
wraps the same plant as a Gymnasium `Env` so that off-the-shelf algorithms can be pointed at
it. [`inflated_rl.py`](code/reinforcement_learning/inflated_rl.py) trains PPO on it for 100k
steps and renders the result.

---

## Running the experiments

All from `code/`, after activating the virtual environment.

| Command | Question it answers | Figure |
| --- | --- | --- |
| `python -m evaluation.policy_gradient_evaluation` | How close do the model-free solvers get to the model-based one, and to the optimum, on a clean full-feedback problem? | 6-panel: trajectories, cost, metadata |
| `python -m evaluation.noise_evaluation` | How does each noise channel, in isolation, degrade the natural policy gradient? | one 6-panel figure per channel |
| `python -m evaluation.inflate_evaluation` | Under inflation, does the low-rank solver beat the unstructured one — and does that hold under noise? | one 6-panel figure per condition, incl. rank |
| `python -m evaluation.spi_evaluation` | Does S-PI actually reduce the rank of the gain, and what does that cost? | 4-panel: rank, cost, trajectories |

Model-free runs are not fast: `m=100` trajectories of `l=200` steps means 40 000 simulation
steps per iteration, and S-PI draws two gradient estimates per iteration.

---

## References

The PDFs live outside this repository, in a sibling `Material/` folder.

| Paper | Implemented in | Local copy |
| --- | --- | --- |
| Park, Rossi, Wen, Wu, Zhao — [*Structured Policy Iteration for Linear Quadratic Regulator*](https://arxiv.org/abs/2007.06202), ICML 2020 | [`structured_policy_iteration.py`](code/lqr_solvers/structured_policy_iteration.py) — the whole file | `Material/StructuredPolicyIteration.pdf` |
| Fazel, Ge, Kakade, Mesbahi — [*Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator*](https://arxiv.org/abs/1801.05039), ICML 2018 | [`mb_policy_gradient.py`](code/lqr_solvers/mb_policy_gradient.py) (gradient, p. 5) and [`mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) (cost convention, §D.3) | `Material/PolicyOptimizationLQR.pdf` |
| Two-point gradient estimator | [`mf_policy_gradient.py`](code/lqr_solvers/mf_policy_gradient.py) — `estimate_gradient` | `Material/TwoPointEstimator.pdf` |

---

## Code Issues

- `calculate_gradient_model_based_discrete` and `estimate_gradient` are annotated
  `-> np.ndarray` but return tuples.
- `caculate_next_K` and `caculate_next_K_mf` are misspelled.
- `spi_evaluation.py` hard-codes `dt=0.01` when building its systems, independently of the
  `dt` passed to the experiment class.
