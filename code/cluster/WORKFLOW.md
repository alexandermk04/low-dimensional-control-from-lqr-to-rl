# Running the experiments

Every number, figure and table in the thesis comes from the experiment scripts in `code/experiments/`. Each script
trains a grid of runs and stores them in `code/experiment_results/<experiment>/`. Then a figure script in
`code/experiments/figures/` turns the stored runs into the figures and tables in `thesis/generated/<experiment>/`.

The state-space experiments are cheap, but the RL and camera experiments add up to thousands of CPU hours (§2). So the
experiments run on the TUHH HPC cluster, and the figures are made on the local machine:

- **Code travels through git** (GitLab on collaborating.tuhh.de).
- **Results travel through `cluster/sync.sh`** (rsync).
- **Each experiment runs as a SLURM job array**, submitted with `cluster/submit.sh`.

Each run is independent and single-threaded. On the cluster a whole experiment runs in parallel, so it takes about as
long as its slowest shard plus the queue time. A single run is *not* faster there; an M-series Mac core is about as
fast as an EPYC core, or faster.

Placeholders: `<rz-user>` is your RZ account name. `$WORK` is your `/work` directory on the cluster (5 TB quota;
`$HOME` has only 20 GB). All commands run from `code/` unless stated otherwise.

---

## 0. One-time setup

### 0.1 Local environment (Python ≥ 3.12)
```bash
# local machine, in the repository root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd code
python -c "import numpy, scipy, control, sippy_unipi, torch, stable_baselines3, gymnasium, tqdm; print('ok')"
```
Locally you need this for the figure scripts, for the helper scripts in `cluster/`, and for any experiment you run
locally (`python -m experiments.<experiment>`; on macOS it keeps the machine awake while it runs).

### 0.2 SSH from the local machine to the cluster
```bash
# local machine
ssh-keygen -t ed25519 -C "local -> tuhh hpc"       # accept the default path, set a passphrase
```
Add to `~/.ssh/config`:
```
Host hpc
    HostName hpclogin.rz.tuhh.de
    User <rz-user>
    IdentityFile ~/.ssh/id_ed25519
    AddKeysToAgent yes
    UseKeychain yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```
```bash
ssh-copy-id hpc          # asks for your RZ password one last time
ssh hpc                  # should now log in without a password
```
Off campus, connect to the TUHH VPN first; the login nodes are only reachable from the TUHH network. If ssh warns
about the host key, add the known_hosts entry from the RZ page "Access and first steps". `UseKeychain` is macOS only;
drop it on Linux.

`ControlMaster` reuses one connection for up to 4 h, so `sync.sh` and repeated `ssh hpc ...` don't re-authenticate.

### 0.3 Git access from the cluster
```bash
# cluster (ssh hpc)
ssh-keygen -t ed25519 -C "tuhh hpc -> gitlab"
cat ~/.ssh/id_ed25519.pub
```
Paste the key in GitLab: collaborating.tuhh.de → avatar → *Preferences* → *SSH Keys* → *Add new key*. Then:
```bash
ssh -T git@collaborating.tuhh.de       # "Welcome to GitLab, @..."
cd $WORK
git clone git@collaborating.tuhh.de:ccf6681/low-dimensional-control-from-lqr-to-rl.git
```
The path is now `$WORK/low-dimensional-control-from-lqr-to-rl`, which `sync.sh` expects.

### 0.4 Python environment on the cluster
The cluster has no Python ≥ 3.12 module (`anaconda/2023.07-1` has 3.11). Use the Anaconda module's `conda` only to get
a bare Python 3.12, and build the venv from it:
```bash
# cluster
module load anaconda/2023.07-1
export CONDA_PKGS_DIRS=$WORK/conda/pkgs         # keep conda's package cache off $HOME

# bare Python 3.12 from conda-forge (avoids Anaconda's default channel and its license terms)
conda create -y -p $WORK/conda/py312 -c conda-forge --override-channels python=3.12
$WORK/conda/py312/bin/python3 --version          # 3.12.x

$WORK/conda/py312/bin/python3 -m venv $WORK/venvs/thesis
source $WORK/venvs/thesis/bin/activate
pip install --upgrade pip
cd $WORK/low-dimensional-control-from-lqr-to-rl
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch, no 2 GB CUDA download
pip install -r requirements.txt
```
The `torch` line comes first on purpose: `requirements.txt` lists `torch`, and installing it from the CPU index
beforehand stops pip from pulling the ~2 GB CUDA wheel.

No `conda init` / `conda activate` is needed; ignore conda if it suggests it. If `conda create` fails, install
Miniforge instead and create the same env with it:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $WORK/miniforge
$WORK/miniforge/bin/conda create -y -p $WORK/conda/py312 python=3.12
```
then continue with the `venv` line above. Test it with the `import` line from §0.1.

The venv points directly at `$WORK/conda/py312/bin/python3`, so jobs need **no `module load`** and nothing in
`~/.bashrc`; `submit.sh` only activates the venv. **Don't delete `$WORK/conda/py312`**, the venv breaks without it.

The scripts in `cluster/` are committed as executable. If they aren't after cloning, run `chmod +x cluster/*.sh` once.

### 0.5 Partition limits
Queried with `sinfo -s` and `sinfo -o "%P %l %c %m %G" | sort -u`:

| Partition | Max time | Nodes | Cores / memory per node | Use |
|---|---|---|---|---|
| `smp` | 21 days | 180 (`g161-228`, `n001-112`) | 24+ cores, 128 GB+ | **all our jobs** |
| `mpp` | 21 days | 112 (`n001-112`, subset of `smp`) | 64 cores, 385 GB+ | multi-node MPI, not needed |
| `gpu` | 7 days | 5 (`u008-012`) | 64–128 cores, 4× A100/H100/H200 | not needed |
| `routing` (default), `ether`, `ib` | – | 0 | – | no own nodes |

- `submit.sh` requests `--partition=smp` explicitly (override with `PARTITION=... cluster/submit.sh ...`). The default
  `routing` partition has no nodes of its own.
- **The 21-day limit never binds**, even for the camera runs (about 4 days each). Still request a realistic time of
  about 1.5× the expected one: short jobs are backfilled into gaps and start sooner, and a hanging run doesn't block a
  slot for weeks.
- **Memory and cores aren't a constraint either.** Every `smp` node has at least 24 cores and 128 GB, so the largest
  request (4 cores, 16 GB) fits on any node.
- **More shards are cheap**, as long as there is no per-user job cap. If `squeue --me` shows many tasks pending with
  reason `QOSMaxJobsPerUserLimit` / `AssocGrpCPULimit`, more shards won't help.

---

## 1. The loop for every experiment

```text
Local:    edit settings.py   →  git commit + git push
Cluster:  git pull           →  cluster/submit.sh ...          (ssh hpc)
          squeue / logs      →  wait
Local:    cluster/sync.sh pull <experiment>  →  figure script  →  thesis/generated/<experiment>/
```

Concretely:
```bash
# local
git add -A && git commit -m "..." && git push

# cluster
ssh hpc
cd $WORK/low-dimensional-control-from-lqr-to-rl && git pull
cd code
source $WORK/venvs/thesis/bin/activate
cluster/submit.sh <experiment> <shards> <time> [cpus] [mem] [extra sbatch args]

# local, once finished
cluster/sync.sh pull <experiment>
python -m experiments.figures.<figure script>      # see §3.6
```

**Rules that matter:**
- **`settings.py` must be identical on both sides.** Each experiment stores the settings it used in its manifest, and
  a mismatch raises `SettingsChangedError`. Git takes care of this as long as you commit before pulling on the cluster.
- **A value stored in the manifest of a pending or running experiment must not change**, or its next task raises
  `SettingsChangedError`. Only push a new value once every experiment that stores it has finished. If you change a
  value on purpose for an experiment that already has runs, update its manifest with
  `python cluster/patch_manifest.py <experiment> <entry>` (dry run, then again with `--write`).
- **Finished runs are always skipped.** Shard `i` of `N` trains the grid entries `i, i+N, i+2N, ...` (grid order = the
  loop order in `build_specs`) and skips the ones already stored. So a timed-out task is fixed by resubmitting the same
  command. A run that was cut off restarts from zero, though; only finished runs are kept.
- **Don't run the same experiment locally and on the cluster at the same time.** The runs aren't lost, but they're
  computed twice. When both sides have runs, `pull` merges them: run directories are unique, and the newest record per
  run wins. To **continue on the cluster** an experiment started locally, `cluster/sync.sh push <experiment>` first,
  then submit.
- **`sync.sh pull` never deletes.** To start an experiment over, delete `experiment_results/<experiment>/` on both
  sides.
- **Don't run experiments on the login node** (`python -m experiments...` directly). Always use `submit.sh`.

### Choosing `<shards>` and `<time>`
- **Shards:** at most the number of grid entries. More shards means more parallel work, but also more jobs in the queue.
- **Time:** (sum of the run times in the largest shard) × 1.5. §2 lists the run times, §3 the values that worked.
- Count grid entries with `python -c "from experiments.<experiment> import build_specs; print(len(build_specs()))"`.
  The calibration and noise experiments have no `build_specs`. They shard over noise channels (calibration) or over
  the source runs they score (noise experiments).

### Monitoring
```bash
squeue --me                                    # your jobs
squeue --me --start                            # estimated start times
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f cluster/logs/<experiment>_<jobid>_<task>.out
scancel <jobid>                                # whole array;  <jobid>_<task> for one task
```

**Chaining:** `submit.sh` passes extra arguments on to `sbatch`. With `--parsable` it prints only the job id, and
`--dependency=afterok:<jobid>` starts a job once another has finished:
```bash
jid=$(cluster/submit.sh ch4_true_model 12 01:00:00 1 4G --parsable)
cluster/submit.sh ch4_true_model_noise 6 01:00:00 1 4G --dependency=afterok:$jid
```
`afterok` needs *every* task to succeed. If one times out, the dependent job stays pending with
`DependencyNeverSatisfied`: `scancel` it, resubmit the timed-out experiment, and submit the dependent job again. Only
chain jobs that don't need a value pasted into `settings.py` in between.

---

## 2. Expected run times

Per run, from the `total_time` of the stored records (October 2026, on the cluster with the cores listed in §3).
`total_time` counts only the training loop: it leaves out the test protocol, the learning-curve probes, and for the
RL solvers the encoder pretraining and the N4SID identification. So the wall time of a run is longer. The last column
is the CPU time of the whole experiment, i.e. roughly how long it would take on one local core.

| Experiment | Per run (median / max) | Whole experiment |
|---|---|---|
| `appendix_sample_size` | `mf_npg`: base 0.3 / 1.2 h, small 0.3 / 1.0 h, medium 0.5 / 1.7 h, **large 3.7 / 9.5 h** | 286 h |
| `appendix_identification_length` | `mb_est`: seconds, up to 9 min at *large* | 3 h |
| `appendix_learning_rate` | `mb_pg`, `mb_est`: seconds, up to 34 min at *large*; `mf_npg`: 0.5–0.8 h, **large 5.0 / 7.2 h** | 309 h |
| `appendix_spi_lambda` | `spi`, `spi_est`: under 2 min | 1 h |
| `appendix_spi_mf` | `spi_mf`: 5 min (base) to 25 min (*large*) | 141 h |
| `appendix_rl_learning_rate` | small 3–8 h, medium 4–12 h, large 6–24 h; slowest `ppo_bottleneck` and `ppo_fc_extractor` | 2030 h |
| `appendix_rl_convergence` | as above, 3 seeds | 370 h |
| `ch4_true_model` | seconds | 0.2 h |
| `ch4_identified_model` | `mf_npg` / `spi_mf`: up to 1 h, **large 4 / 6 h**; the rest seconds | 108 h |
| `ch5_inflation_sizes` | small 1–8 h, medium 2–17 h, large 7–22 h; slowest `ppo_fc_extractor` | 1210 h |
| `ch6_solver_comparison` | `ppo_mlp` 8 / 10 h, `ppo_encoder_scalable` 23 / 24 h, `ppo_fc_extractor` 46 / 55 h, **`ppo_bottleneck` 56 / 85 h** | 1310 h |
| `ch6_perception` | `ppo_pixel_estimator` 17 / 19 h, `ppo_pixel_posvel` 23 / 24 h, **`ppo_pixel_fc` 83 / 85 h, `ppo_pixel_ortho` 85 / 94 h** | 1960 h |
| `appendix_calibrate_noise` | not timed; one bisection of the baseline per channel, the 6 h in §3.1 was enough | – |
| `appendix_calibrate_pixel_noise` | ≈ 1.5–2 h per level (bisection, each step 10 seeds × 100 episodes of `oracle_coarse`), so **6–8 h per channel** | ≈ 20 h |
| `ch4_true_model_noise`, `ch4_noise` | ≈ 1.6 s (base) to 3 s (*large*) per noise cell | < 1 h |
| `ch5_noise` | per source policy, 21 cells | < 1 h per shard |
| `ch6_noise` | ≈ 65–80 s per policy and cell, so ≈ 25 min per policy | ≈ 17 h |

The RL budgets are set by `RL_MAX_TIMESTEPS` in `settings.py` (small 10 M, medium 15 M, large 30 M env steps; the
camera runs 10 M agent decisions, i.e. 300 M plant steps at 30× action repeat). The RL runs have no stopping rule
and always train to that cap (`appendix_rl_convergence` shows why), with the learning rate annealing to zero at it.
So their time is predictable, and a run that is cut off loses everything: **request generously.**

---

## 3. Full progression, from an empty `experiment_results/`

The experiments depend on each other in two ways: through **values in `settings.py`** that one experiment chooses and
later ones read, and through **stored policies** that the noise experiments score. That gives four waves. Within a
wave everything can be submitted at once.

```text
Wave A   sample size, identification length, noise calibrations          (no dependencies)
            │  CHOSEN_SAMPLE_CELL, CHOSEN_IDENTIFICATION, NOISE_STD, CH6_NOISE_STD
            ▼
Wave B   hyperparameter sweeps (PG, S-PI, model-free S-PI, PPO)
            │  SIZE_HYPERPARAMS
            ▼
Wave C   main experiments: Ch. 3 (ch4_*), Ch. 4 (ch5_*), Ch. 5 (ch6_*), RL convergence
            │  stored policies
            ▼
Wave D   noise experiments, one per main experiment
```

(Code chapters `ch4`/`ch5`/`ch6` are thesis chapters 3/4/5.)

**About the values in `settings.py`.** The repository ships with the values the thesis uses, each marked with a
*Filled from* comment naming the script that chooses it. After each wave, run that figure script: it prints its
choice. If the choice matches the value in `settings.py`, go on. If not, paste the printed value, commit, push and
`git pull` on the cluster before submitting the next wave. To force a wave to wait for its inputs, set the values to
`None`: an experiment that needs a missing value stops with a `Run: ...` error instead of using a stale one.

### 3.1 Wave A: budgets and noise levels
No dependencies; submit all four together.

| Experiment | Chooses (paste into `settings.py`) | Figure script that prints it |
|---|---|---|
| `appendix_sample_size` | `CHOSEN_SAMPLE_CELL` (rollout length, trajectory count per size) | `appendix_sample_size_figures` |
| `appendix_identification_length` | `CHOSEN_IDENTIFICATION` (trajectory length, Hankel horizon per size) | `appendix_identification_figures` |
| `appendix_calibrate_noise` | `NOISE_STD` (state-space noise, 4 channels × 5 levels) | `appendix_calibrate_noise_figures` |
| `appendix_calibrate_pixel_noise` | `CH6_NOISE_STD` for `control`, `parametric`, `disturbance` | `appendix_calibrate_pixel_noise_figures` |

```bash
# cluster
cluster/submit.sh appendix_sample_size 40 24:00:00 1 4G
cluster/submit.sh appendix_identification_length 30 02:00:00 1 4G
cluster/submit.sh appendix_calibrate_noise 4 06:00:00 1 4G
cluster/submit.sh appendix_calibrate_pixel_noise 3 12:00:00 1 4G
```
`appendix_sample_size` and `appendix_identification_length` train at the fixed learning rate `GRADIENT_LR` in their
script, because the learning-rate sweep (wave B) needs the budget they choose. The camera's `measurement` levels in
`CH6_NOISE_STD` are not calibrated: that noise acts on the image, and its levels are set by hand.

### 3.2 Wave B: hyperparameters
Needs `CHOSEN_SAMPLE_CELL` and `CHOSEN_IDENTIFICATION` from wave A.

| Experiment | Chooses (into `SIZE_HYPERPARAMS`) | Figure script that prints it |
|---|---|---|
| `appendix_learning_rate` | `gradient_lr` of `mb_pg`, `mb_est`, `mf_npg` | `appendix_learning_rate_figures` |
| `appendix_spi_lambda` | `lambda_` of `spi`, `spi_est` | `appendix_spi_lambda_figures` |
| `appendix_spi_mf` | `lambda_`, `stepsize` of `spi_mf` | `appendix_spi_mf_figures` |
| `appendix_rl_learning_rate` | `learning_rate` of every PPO solver at small, medium, large | `appendix_rl_learning_rate_figures` |

```bash
cluster/submit.sh appendix_learning_rate 40 24:00:00 1 4G
cluster/submit.sh appendix_spi_lambda 30 02:00:00 1 4G
cluster/submit.sh appendix_spi_mf 50 08:00:00 1 4G
cluster/submit.sh appendix_rl_learning_rate 270 72:00:00 2 8G
```
**The camera learning rates are set by hand.** The camera size is far too large to sweep, so the `pixel` entries of
`ppo_bottleneck`, `ppo_encoder_scalable`, `ppo_mlp` and `ppo_fc_extractor` are their *large* rate scaled by
√(200 / 18432) ≈ 0.104, the same square-root rule by which `RL_LEARNING_RATES` scales the sweep grid between sizes.
Chapter 5's camera solvers borrow these rates through `CH6_LEARNING_RATE_FROM`. Update them by hand whenever the
*large* rates change.

### 3.3 Wave C: main experiments
Needs every value from waves A and B. All six are independent of each other.

| Experiment | Thesis place |
|---|---|
| `ch4_true_model` | Ch. 3, true model |
| `ch4_identified_model` | Ch. 3, unknown model |
| `ch5_inflation_sizes` | Ch. 4 |
| `appendix_rl_convergence` | App. B, why the RL runs have no stopping rule |
| `ch6_solver_comparison` | Ch. 5 §5.1, the Ch. 4 solvers on the camera |
| `ch6_perception` | Ch. 5, the perception-based solvers |

```bash
jid_true=$(cluster/submit.sh ch4_true_model 12 01:00:00 1 4G --parsable)
jid_ident=$(cluster/submit.sh ch4_identified_model 30 12:00:00 1 4G --parsable)
jid_rl=$(cluster/submit.sh ch5_inflation_sizes 210 72:00:00 2 8G --parsable)
cluster/submit.sh appendix_rl_convergence 54 72:00:00 2 8G
cluster/submit.sh ch6_solver_comparison 50 7-00:00:00 2 16G
jid_cam=$(cluster/submit.sh ch6_perception 50 7-00:00:00 4 16G --parsable)
```
The RL and camera experiments use one shard per run, since each run takes hours to days. `ch6_perception` takes
about four days, so it decides when the whole progression finishes.

After `ch4_noise` (wave D), look at its figures and set `CHOSEN_NOISE_LEVEL` and `CHOSEN_NOISE_CHANNEL`, the level and
channel the per-channel and per-level comparisons of Ch. 3 are drawn at. They only change figures, not runs.

### 3.4 Wave D: noise
Each noise experiment scores the stored policies of one main experiment under every noise channel and level, at test
time only. It needs that experiment finished and `NOISE_STD` / `CH6_NOISE_STD` from wave A, so it can be chained
directly to wave C (the job ids from §3.3):

| Experiment | Scores the policies of | Thesis place |
|---|---|---|
| `ch4_true_model_noise` | `ch4_true_model` | Ch. 3 |
| `ch4_noise` | `ch4_identified_model` | Ch. 3 |
| `ch5_noise` | `ch5_inflation_sizes` (at *large*) | Ch. 4 |
| `ch6_noise` | `ch6_perception` (without the oracle, which needs the true state) | Ch. 5 |

```bash
cluster/submit.sh ch4_true_model_noise 6 01:00:00 1 4G --dependency=afterok:$jid_true
cluster/submit.sh ch4_noise 12 01:00:00 1 4G --dependency=afterok:$jid_ident
cluster/submit.sh ch5_noise 15 01:00:00 2 8G --dependency=afterok:$jid_rl
cluster/submit.sh ch6_noise 40 02:00:00 2 8G --dependency=afterok:$jid_cam
```
If a wave-C task timed out, its noise job stays pending (see *Chaining* in §1). Resubmit the main experiment, then the
noise job; a noise job that ran before all its sources existed can simply be submitted again, since it skips the
cells it already scored.

### 3.5 Overview

| Wave | Experiments | Needs | Takes about |
|---|---|---|---|
| A | `appendix_sample_size`, `appendix_identification_length`, `appendix_calibrate_noise`, `appendix_calibrate_pixel_noise` | – | 1 day |
| B | `appendix_learning_rate`, `appendix_spi_lambda`, `appendix_spi_mf`, `appendix_rl_learning_rate` | A pasted | 1–2 days |
| C | `ch4_true_model`, `ch4_identified_model`, `ch5_inflation_sizes`, `appendix_rl_convergence`, `ch6_solver_comparison`, `ch6_perception` | B pasted | 4–5 days |
| D | `ch4_true_model_noise`, `ch4_noise`, `ch5_noise`, `ch6_noise` | its C experiment | hours |

So about 1.5 weeks from an empty `experiment_results/` to the last figure, without queue time.

### 3.6 Figures, tables and the thesis
After each experiment, locally: `cluster/sync.sh pull <experiment>`, then its figure script. Each one writes to
`thesis/generated/<experiment>/{figures,tables}/`.

| Experiment | Figure script (`python -m experiments.figures.<…>`) |
|---|---|
| `appendix_sample_size` | `appendix_sample_size_figures` |
| `appendix_identification_length` | `appendix_identification_figures` |
| `appendix_calibrate_noise` | `appendix_calibrate_noise_figures` |
| `appendix_calibrate_pixel_noise` | `appendix_calibrate_pixel_noise_figures` |
| `appendix_learning_rate` | `appendix_learning_rate_figures` |
| `appendix_spi_lambda` | `appendix_spi_lambda_figures` |
| `appendix_spi_mf` | `appendix_spi_mf_figures` |
| `appendix_rl_learning_rate` | `appendix_rl_learning_rate_figures` |
| `appendix_rl_convergence` | `appendix_rl_convergence_figures` |
| `ch4_true_model` | `ch4_true_model_figures` |
| `ch4_identified_model` | `ch4_identified_model_figures` |
| `ch4_true_model_noise` | `ch4_true_model_noise_figures` |
| `ch4_noise` | `ch4_noise_figures` |
| `ch5_inflation_sizes` | `ch5_inflation_figures` |
| `ch5_noise` | `ch5_noise_figures` |
| `ch6_solver_comparison` | `ch6_solver_comparison_figures` |
| `ch6_perception` | `ch6_perception_figures` |
| `ch6_noise` | `ch6_noise_figures` |

Then rebuild the thesis with `latexmk` in `thesis/`.

The other scripts in `cluster/` (`recount_policy_size.py`, `recount_runaway.py`, `reprobe_control.py`) add fields
to records written before the experiments stored them. Fresh runs store `policy_complexity_frozen`,
`runaway_episodes` and `r2_control` themselves, so a fresh progression doesn't need them.
