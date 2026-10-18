# Running experiments on the TUHH HPC cluster

How it works: **code travels through git** (GitLab on collaborating.tuhh.de), **results travel through `cluster/sync.sh`** (rsync), and **each experiment runs as a SLURM job array** via `cluster/submit.sh`.

Why it helps: each run is independent and single-threaded, so on the Mac they run one after another on 1 of 10 cores. On the cluster a whole experiment runs in parallel, so it takes about as long as its slowest single run plus queue time. A single run is *not* faster there; the M5 core is about as fast as an EPYC core, or faster.

Placeholders: `<rz-user>` is your RZ account name. Everything below `$WORK` on the cluster refers to your `/work` directory (5 TB quota; `$HOME` has only 20 GB).

---

## 0. One-time setup

### 0.1 SSH from the Mac to the cluster
```bash
# on the Mac
ssh-keygen -t ed25519 -C "mac -> tuhh hpc"          # accept the default path, set a passphrase
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
Off campus, connect to the TUHH VPN first; the login nodes are only reachable from the TUHH network. If ssh warns about the host key, add the known_hosts entry from the RZ page "Access and first steps".

`ControlMaster` reuses one connection for up to 4 h, so `sync.sh` and repeated `ssh hpc ...` don't re-authenticate.

### 0.2 Git access from the cluster
```bash
# on the cluster (ssh hpc)
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

### 0.3 Python environment (Python ≥ 3.12)
The cluster has no Python ≥ 3.12 module (`anaconda/2023.07-1` has 3.11, `anaconda/2020.11` has 3.8). Use the Anaconda module's `conda` only to get a bare Python 3.12, and build the venv from it:
```bash
# on the cluster
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
The `torch` line comes first on purpose: `requirements.txt` lists `torch`, and installing it from
the CPU index beforehand stops pip from pulling the ~2 GB CUDA wheel. Everything else the code needs
(`tqdm`, `control`) is now in `requirements.txt`, so the extra `pip install` step from the earlier
version of this file is no longer needed.

No `conda init` / `conda activate` is needed; ignore conda if it suggests it. If `conda create` fails, install Miniforge instead and create the same env with it:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $WORK/miniforge
$WORK/miniforge/bin/conda create -y -p $WORK/conda/py312 python=3.12
```
then continue with the `venv` line above.

Test it:
```bash
cd code
python -c "import numpy, scipy, control, sippy_unipi, torch, stable_baselines3, gymnasium, tqdm; print('ok')"
```

The venv points directly at `$WORK/conda/py312/bin/python3`, so jobs need **no `module load`** and nothing in `~/.bashrc`; `submit.sh` only activates the venv. **Don't delete `$WORK/conda/py312`**, the venv breaks without it.

### 0.4 Partition limits
Queried with `sinfo -s` and `sinfo -o "%P %l %c %m %G" | sort -u`:

| Partition | Max time | Nodes | Cores / memory per node | Use |
|---|---|---|---|---|
| `smp` | 21 days | 180 (`g161-228`, `n001-112`) | 24+ cores, 128 GB+ | **all our jobs** |
| `mpp` | 21 days | 112 (`n001-112`, subset of `smp`) | 64 cores, 385 GB+ | multi-node MPI, not needed |
| `gpu` | 7 days | 5 (`u008-012`) | 64–128 cores, 4× A100/H100/H200 | not needed (see step G) |
| `routing` (default), `ether`, `ib` | – | 0 | – | no own nodes |

What this means:
- `submit.sh` requests `--partition=smp` explicitly (override with `PARTITION=... cluster/submit.sh ...`). The default `routing` partition has no nodes of its own.
- **Time limits never bind.** Even the largest request below (a few hours per task) is far under 21 days. Still request a realistic time with about 1.5× safety rather than days: short jobs are backfilled into gaps and start sooner, and a hanging run doesn't block a slot for weeks.
- **Memory and cores aren't a constraint either.** Every `smp` node has at least 24 cores and 128 GB, so the largest request (4 cores, 16 GB) fits on any node.
- **More shards are cheap.** At query time about 30 `smp` nodes (> 700 cores) were idle, so a 40-task array of single-core jobs usually starts right away. What limits parallelism is any per-user job cap, not the hardware: if `squeue --me` shows many tasks pending with reason `QOSMaxJobsPerUserLimit` / `AssocGrpCPULimit`, more shards won't help.

### 0.5 Make the scripts executable on the cluster
They're executable locally already. Once they're committed, git keeps that. Otherwise run `chmod +x code/cluster/*.sh` once after cloning.

---

## 1. The loop for every experiment

```text
Mac:      edit settings.py   →  git commit + git push          (you do this)
Cluster:  git pull           →  cluster/submit.sh ...          (ssh hpc)
          squeue / logs      →  wait
Mac:      cluster/sync.sh pull <experiment>  →  figures/tables locally
```

Concretely:
```bash
# Mac
git add -A && git commit -m "..." && git push

# Cluster
ssh hpc
cd $WORK/low-dimensional-control-from-lqr-to-rl && git pull
cd code
source $WORK/venvs/thesis/bin/activate
cluster/submit.sh <experiment> <shards> <time> [cpus] [mem] [extra sbatch args]

# Mac, once finished
code/cluster/sync.sh pull <experiment>
```

**Rules that matter:**
- **`settings.py` must be identical on both sides.** The manifest check raises `SettingsChangedError` otherwise. Git takes care of this as long as you commit before pulling on the cluster.
- **Don't run the same experiment on the Mac and on the cluster at the same time.** The runs aren't lost, but they're computed twice. When both sides have runs, `pull` merges them: run directories are unique, and the newest record per run wins.
- To **continue on the cluster** an experiment that was started on the Mac: `code/cluster/sync.sh push <experiment>` first, then submit. Finished runs are skipped.
- **Timeouts are harmless.** Resubmit the same command; finished runs are skipped.
- **Don't run experiments on the login node** (`python -m experiments...` directly). Always use `submit.sh`.

### How to choose `<shards>` and `<time>`
- Shard `i` of `N` trains grid entries `i, i+N, i+2N, ...` (grid order = the loop order in `build_specs`). Entries that are already finished are skipped.
- **Shards:** at most the number of grid entries. More shards means more parallel work, but also more jobs waiting in the queue.
- **Time:** (sum of run times in the largest shard) × 1.5. The per-run times below were measured on the Mac.
- Count grid entries with: `python -c "from experiments.<exp> import build_specs; print(len(build_specs()))"`

### Monitoring
```bash
squeue --me                                    # your jobs
squeue --me --start                            # estimated start times
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f cluster/logs/<experiment>_<jobid>_<task>.out
scancel <jobid>                                # whole array;  <jobid>_<task> for one task
```
Chain experiments with `--dependency=afterok:<jobid>` as an extra argument to `submit.sh`.

---

## 2. Measured run times (one core)

Taken from the `total_time` of the stored records (September 2026, **before** the cost moved to the controller space; refresh after wave A). `total_time` leaves out the probe overhead, so wall time is about 1.5–2× these numbers.

| Solver | Per run |
|---|---|
| `mf_npg` / `spi_mf` at (800, 100) | ≈ 0.7 h (base), 0.8 h (small), 1.1 h (medium), **3.4 h (large)** |
| `baseline`, `mb_pg`, `spi`, `ce_dare`, `mb_est`, `spi_est` | < 20 s (plus the test protocol); since the cost moved to the controller space, `mb_pg`/`spi` solve p-dim Lyapunov equations: ≈ 7 s / 13 s at *large* (Mac, October 2026) |
| ch5 sizes, per 1 M steps | `ppo_linear` ≈ 1 000 s, `ppo_encoder` ≈ 750 s, `ppo_encoder_scalable` / `ppo_mlp` ≈ 500 s |
| ch5 pixel, per 1 M steps | `ppo_linear` ≈ 2.5 h, `ppo_mlp` ≈ 0.7–1.2 h, `ppo_encoder_scalable` ≈ 1.6–2.2 h, **`ppo_bottleneck` ≈ 5.5 h** |
| `ppo_fc_extractor` | not timed yet; same 512–256–256 trunk as `ppo_bottleneck`, so expect the same time |
| `ppo_pixel_*` (ch6) | ≈ 1–1.3 h per 1 M agent steps, including probes and pretraining |
| noise cell, state space (`ch4_noise`, `ch4_true_model_noise`) | ≈ 1.6 s at base, 3 s at large (test protocol + figure trajectory) |
| noise calibration, camera env (`appendix_calibrate_pixel_noise`) | ≈ 1.4 h per level (`measurement`) to ≈ 2 h per level (`control`, `disturbance`) on the cluster (bisection, each step 10 seeds × 100 episodes of `oracle_coarse`), so **6–8 h per channel** (job 6432675). Since 10-07 only `control`, `parametric`, `disturbance` are calibrated; measurement noise acts on the image and its levels are set by hand |
| noise cell, camera env (`ch6_noise`) | ≈ 65–80 s per `ppo_pixel_*` policy, `oracle_coarse` is no longer scored (100 episodes × 800 agent steps; measured with untrained policies of the same architecture) |

`RL_MAX_TIMESTEPS` is a cap per inflation size (small 10 M, medium 15 M, large 30 M, pixel 10 M env steps; pixel was 30 M until 2026-10-14, see §5), and the learning rate anneals to zero at it. Chapter 6 uses `RL_MAX_TIMESTEPS["pixel"]` as well, counted in agent decisions (× 30 action repeat = 300 M plant steps); its learning curves are in decisions too (since 2026-10-14). On the cluster (2 cores), wall time was about 2× `total_time`: the slowest `ppo_encoder` runs needed ≈ 5.9 h for 10 M steps, and every `ppo_bottleneck` / `ppo_encoder_scalable` run hit the 6 h limit at 10 M. At 30 M (*large*), `ppo_encoder` in `appendix_rl_learning_rate` took 15–19 h wall time (job 6430657, October 2026). **Request generously:** a run that times out restarts from zero on resubmission (only finished runs are skipped).

---

## 3. To do — restart from zero (cost in the controller space)

Status as of **2026-10-06**. The stage cost is now charged in the controller's space (`Q`: p×p on the clean `U₁x`,
`R`: q×q on `v`), see `Planning/Inflated_Cost_Restructure.md`. **Every earlier result is invalid** and all
experiments are rerun in the order below. Each wave starts once the values it needs have been pasted into
`settings.py`, committed, pushed and pulled on the cluster. All values marked *Filled from* in `settings.py` are
reset to `None` (and `SIZE_HYPERPARAMS` is empty, with no global fallback for `gradient_lr`, `lambda_`, `stepsize`),
so a wave started too early stops with a "Run: ..." error instead of using a stale value. The sweep grids are kept
as they were, centred on the pre-restructure choices. Wave A runs `mf_npg` / `mb_est` at a fixed learning rate
(`GRADIENT_LR` in `appendix_sample_size.py` / `appendix_identification_length.py`), because the learning-rate
sweep needs the budget Wave A chooses.

**0. Remove all old results** (once, before wave A), on the Mac **and** on the cluster (`sync.sh pull` never deletes):
```bash
# Mac (in code/) and cluster (in $WORK/low-dimensional-control-from-lqr-to-rl/code)
rm -rf experiment_results/*
```
Also delete the old generated figures/tables of the thesis when the new ones are rendered.

| Wave | Experiment | Needs | Produces (paste into `settings.py`) | Shards / time / cpus / mem |
|---|---|---|---|---|
| A | `appendix_sample_size` | – | `CHOSEN_SAMPLE_CELL` | 240 entries; 40 / 12 h / 1 / 4G |
| A | `appendix_identification_length` | – | `CHOSEN_IDENTIFICATION` | 600 entries; 30 / 6 h / 1 / 4G |
| A | `appendix_calibrate_noise` | – | `NOISE_STD` | 4 / 6 h / 1 / 4G |
| A | `appendix_calibrate_pixel_noise` | – | `CH6_NOISE_STD` (3 channels) | 3 / 12 h / 1 / 4G |
| B | `appendix_learning_rate` | A | PG learning rates (`SIZE_HYPERPARAMS`) | 800 entries; 40 / 12 h / 1 / 4G |
| B | `appendix_spi_lambda` | A | SPI `lambda_` (`SIZE_HYPERPARAMS`) | 480 entries; 30 / 6 h / 1 / 4G |
| B | `appendix_spi_mf` | A | `spi_mf` `lambda_`, `stepsize` (`SIZE_HYPERPARAMS`) | 1000 entries; 50 / 24 h / 1 / 4G |
| B | `appendix_rl_learning_rate` | A | RL `learning_rate` (`SIZE_HYPERPARAMS`) | 225 entries; 225 / 72 h / 2 / 8G |
| C | `ch4_true_model` | B | Ch. 4 track A | 120 entries; 12 / 2 h / 1 / 4G |
| C | `ch4_identified_model` | B | Ch. 4 track B | 30 / 12 h / 1 / 4G |
| C | `appendix_rl_convergence` | B | `CONVERGENCE["rl"]` | 45 / 72 h / 2 / 8G |
| D | `ch4_true_model_noise` | `ch4_true_model` | Ch. 4 noise (true model) | 6 / 1 h / 1 / 4G |
| D | `ch4_noise` | `ch4_identified_model` | Ch. 4 noise | 12 / 1 h / 1 / 4G |
| D | `ch5_inflation_sizes` | C (`CONVERGENCE["rl"]`) | Ch. 5 | 180 / 72 h / 2 / 8G |
| D | `ch6_solver_comparison` | C | Ch. 6 comparison | 40 / 72 h / 2 / 16G |
| D | `ch6_perception` | C | Ch. 6 | 50 / 24 h / 4 / 16G |
| E | `ch5_noise` | `ch5_inflation_sizes` | Ch. 5 noise | 15 / 1 h / 2 / 8G |
| E | `ch6_noise` | `ch6_perception`, `CH6_NOISE_STD` | Ch. 6 noise, without the oracle | 40 / 2 h / 2 / 8G |

Shard counts and times for the experiments that had not been timed on the cluster before (`appendix_sample_size`,
`appendix_identification_length`, `appendix_calibrate_noise`, `appendix_learning_rate`, `appendix_spi_lambda`,
`appendix_spi_mf`, `appendix_rl_learning_rate`, `ch4_true_model`) are estimates from §2. Check them against the first
finished tasks and adjust. Count grid entries with `build_specs` (§1).

**Wave A** (no dependencies, submit together):
```bash
# cluster
cluster/submit.sh appendix_sample_size 40 12:00:00 1 4G
cluster/submit.sh appendix_identification_length 30 06:00:00 1 4G
cluster/submit.sh appendix_calibrate_noise 4 06:00:00 1 4G
cluster/submit.sh appendix_calibrate_pixel_noise 3 12:00:00 1 4G
```
Then on the Mac: `sync.sh pull` each, run its figure script (`python -m experiments.figures.<experiment>_figures`),
paste the printed values into `settings.py`, commit, push, `git pull` on the cluster. Update §2 with the new run times.

**Wave B** (needs A pasted and pulled):
```bash
cluster/submit.sh appendix_learning_rate 40 12:00:00 1 4G
cluster/submit.sh appendix_spi_lambda 30 06:00:00 1 4G
cluster/submit.sh appendix_spi_mf 50 24:00:00 1 4G
cluster/submit.sh appendix_rl_learning_rate 225 72:00:00 2 8G
```
`mf_npg` at *large* diverged from the first step with the old learning rate (0.03) in the restructure smoke test,
so check that the learning-rate grid of `appendix_learning_rate` still covers stable values at *large*.

**Wave C** (needs B pasted and pulled):
```bash
jid4a=$(cluster/submit.sh ch4_true_model 12 02:00:00 1 4G --parsable)
jid4b=$(cluster/submit.sh ch4_identified_model 30 12:00:00 1 4G --parsable)
cluster/submit.sh appendix_rl_convergence 45 72:00:00 2 8G
```

**Wave D** (Ch. 4 noise right after C; the RL runs need `CONVERGENCE["rl"]` pasted and pulled):
```bash
cluster/submit.sh ch4_true_model_noise 6 01:00:00 1 4G --dependency=afterok:$jid4a
cluster/submit.sh ch4_noise 12 01:00:00 1 4G --dependency=afterok:$jid4b
jid5=$(cluster/submit.sh ch5_inflation_sizes 180 72:00:00 2 8G --parsable)
cluster/submit.sh ch6_solver_comparison 40 72:00:00 2 16G
jid6=$(cluster/submit.sh ch6_perception 50 24:00:00 4 16G --parsable)
```

**Wave E**:
```bash
cluster/submit.sh ch5_noise 15 01:00:00 2 8G --dependency=afterok:$jid5
cluster/submit.sh ch6_noise 40 02:00:00 2 8G --dependency=afterok:$jid6
```

**Chaining:** `submit.sh` passes extra arguments to `sbatch`, so `--parsable` makes it print only the job id.
`afterok` needs *every* task to succeed. If one times out, the dependent job stays pending with
`DependencyNeverSatisfied`: `scancel` it, resubmit the timed-out experiment, and submit the dependent job again.

**Settings changes between waves:** a value stored in the manifest of a *pending or running* experiment must not
change, or its next task raises `SettingsChangedError`. Only pull a pasted value once every experiment that stores it
has finished, or first run `python cluster/patch_manifest.py <experiment> <key>` (dry run, then `--write`).

**Figures** (after each wave, on the Mac): `sync.sh pull <experiment>`, then
`python -m experiments.figures.<experiment>_figures`. Rebuild the thesis with `latexmk` in `thesis/`.

---

## 4. Overview

| Wave | Experiments | Status 06.10. |
|---|---|---|
| 0 | delete old results (Mac + cluster) | open |
| A | appendix_sample_size, appendix_identification_length, appendix_calibrate_noise, appendix_calibrate_pixel_noise | open |
| B | appendix_learning_rate, appendix_spi_lambda, appendix_spi_mf, appendix_rl_learning_rate | waits for A |
| C | ch4_true_model, ch4_identified_model, appendix_rl_convergence | waits for B |
| D | ch4_true_model_noise, ch4_noise, ch5_inflation_sizes, ch6_solver_comparison, ch6_perception | waits for C |
| E | ch5_noise, ch6_noise | waits for D |
| §5 A | FC-Extractor + PPO: lr sweep → paste → pixel-rate check → ch5_inflation_sizes → ch5_noise; ch6_solver_comparison at 10 M | sweep and ch6_solver_comparison running, 15.10. |
| §5 B | ch6 protocol (`d30a6b2`): pull → calibrate_pixel_noise + ch6_perception → paste → ch6_noise | open (not pulled on the cluster), 15.10. |
| §5 C | recount policy sizes, figures | after A and B |

Cluster times don't include queue waiting. Timed-out tasks: resubmit the same command; finished runs are skipped.

---

## 5. Reruns after the 2026-10-14 changes (status 2026-10-15)

Two changes from 2026-10-14 each force reruns. They are independent except where a chapter 6 solver borrows
the FC-Extractor's learning rate (§5.3, step A3).

### 5.1 What changed

**A. FC-Extractor + PPO and the policy-size recount** (commits `84f5435`, `c43c25b`, see
`Planning/Improvements/Ch4_Artificial_Inflation.md`, C5-30):

- **Policy size now counts the feature extractor.** `policy_complexity` used to count only the actor's head
  (`mlp_extractor.policy_net` + `action_net`). It now includes the actor's feature extractor and, for
  `ppo_encoder`, the identified encoder $\hat{C}^\top$. The new summary field `policy_complexity_frozen` holds the part
  that PPO did not train (the pretrained `ppo_encoder_scalable` / `ppo_pixel_estimator` / `ppo_pixel_posvel`
  extractors and $\hat{C}^\top$). The policy-size bar plots hatch that part. New runs store both fields. Older runs
  are fixed with `cluster/recount_policy_size.py`, which reads each run's `policy.zip`, so **no training is rerun**.
- **New solver `ppo_fc_extractor` ("FC-Extractor + PPO")**: Bottleneck PPO's trunk without the projection $B$, so
  equally large (327 140 vs. 310 884 parameters at *large*). It is appended to `CH5_SOLVERS` (so it joins
  `appendix_rl_learning_rate`, `appendix_rl_convergence`, `ch5_inflation_sizes`, `ch5_noise`) and to
  `CH6_COMPARISON_SOLVERS`. Its *pixel* learning rate is **borrowed** (`7e-07`, from `ppo_bottleneck`) until its own
  sweep is done. Through `CH6_LEARNING_RATE_FROM`, chapter 6's `ppo_pixel_fc` uses the same borrowed rate.
- `RL_MAX_TIMESTEPS["pixel"]` dropped from 30 M to 10 M (see §5.5). The value is stored in
  every RL manifest, hence the `timesteps` / `rl_timesteps` patches below. None of these experiments train at
  *pixel*, so patching changes nothing for their runs.

Adding the solver changes the manifests of those experiments, so each one needs `patch_manifest.py` before its
next submission (otherwise `SettingsChangedError`). Always patch **on the cluster**: its manifest is the one that
counts, and `sync.sh pull` copies it to the Mac afterwards. Run the dry run first and check that only the listed
entries change.

**B. Chapter 6 protocol aligned with chapters 4/5** (commit `d30a6b2`):

- Horizons are the chapter 4 plant-step settings converted to agent decisions (`/ 30` action repeat, rounded up):
  train and probe 27, test 67, figure 27. Before, the camera environment used `TRAIN_HORIZON` = 800 *decisions*
  for training, probes **and** the test protocol.
- Each `(seed, purpose)` gets its own seeded environment (`_pixel_env`), so every solver of a seed sees the same
  test, probe and figure episodes. Test costs are stored per episode, pretraining data comes from $K_0$ instead
  of the optimal gain, and budget and learning curves are counted in decisions.
- Pixel presets reuse the chapter 4/5 PPO setup (heads [64, 64], log std −2, minibatch 64) and the *pixel* learning
  rate of their counterpart (`CH6_LEARNING_RATE_FROM`). Only γ differs.

**Every chapter 6 camera result made before `d30a6b2` is invalid**: `ch6_perception`, `ch6_noise` and
`appendix_calibrate_pixel_noise`. That includes the calibration of job 6541755 (§5.2). `ch6_solver_comparison` is
not affected: it runs the inflated state-space plant at size *pixel*, not the camera environment.

### 5.2 Status 2026-10-15, 08:00

**The cluster is still on `c43c25b`.** `d30a6b2` (and the thesis-only `203cac1`) has not been pulled there, so
nothing on the cluster runs the new chapter 6 protocol yet.

| Experiment | Job | Status |
|---|---|---|
| `appendix_rl_learning_rate` (45 `ppo_fc_extractor` runs) | 6538190 | *small* and *medium*: 30/30 done. *Large*: 15 tasks (30–44) running for 23 h, limit 72 h |
| `ch6_solver_comparison` at 10 M | 6538235 | `baseline` 10/10 (≈ 1.2 h each), `ppo_mlp` 10/10 (10–17 h); `ppo_bottleneck`, `ppo_encoder_scalable`, `ppo_fc_extractor` (tasks 10–29, 40–49) running for 23 h, limit 6 days |
| `appendix_calibrate_pixel_noise` | 6541755 | Ran on the **old** code. `control`, `disturbance` finished; `parametric` timed out at 12 h after only `weak`, which missed its target after all 60 bisection steps (ratio 1.102 vs. 1.2). Invalid, rerun |
| `ch6_perception` | 6533093 | Old protocol, cancelled 10-14 14:47; its runs are gone from `experiment_results/`. Nothing running |
| `ch6_noise` | – | Old results in `experiment_results/_archive_2026-10-13/`. Waits |
| `ch5_inflation_sizes`, `ch5_noise`, `appendix_rl_convergence` | – | Wait for the FC-Extractor learning rate |

`CH6_NOISE_STD["control"]` in `settings.py` already holds the values of job 6541755 (1.31 / 2.09 / 4.15 / 8.78).
Those come from the old protocol and must be replaced after the rerun.

The `parametric` miss needs a look before the rerun is trusted: a bisection that runs all 60 steps without
hitting 1.2 ± tolerance means the cost ratio jumps across the target, e.g. when episodes start to leave the frame
and pick up the exit penalty. Check the per-level ratios of the new run (step B3).

### 5.3 Order

```text
A  FC-Extractor LR    A1 sweep (running) ─ A2 paste ─ A3 pixel-rate check ─┬─ A4 ch5_inflation_sizes ─ A5 ch5_noise
                                                                           │   (+ optional appendix_rl_convergence)
                                                                           └─ decides whether ppo_pixel_fc / ch6 ppo_fc_extractor rerun
B  ch6 protocol       B1 git pull ─┬─ B2a appendix_calibrate_pixel_noise ─ B3 paste CH6_NOISE_STD ─┐
                                   └─ B2b ch6_perception ──────────────────────────────────────────┴─ B4 ch6_noise  (after A3)
C  at the end         C1 recount policy sizes ─ C2 figures
```

Running in parallel right now: A1 and `ch6_solver_comparison`. B1 and B2 can start **today**; they don't wait for A.
B4 waits for A3, because `ch6_noise` scores the `ppo_pixel_fc` policies and those may be retrained in A3.

### 5.4 Steps

#### B1. Pull the chapter 6 code (cluster, now)
```bash
# Mac: push the thesis commit too, so both sides are on the same HEAD
git push
# cluster
cd $WORK/low-dimensional-control-from-lqr-to-rl && git pull && cd code
source $WORK/venvs/thesis/bin/activate
```
Safe while 6538190 and 6538235 run: every task of both arrays is already running and has imported its code, and
`d30a6b2` changes none of the settings their manifests store.

#### B2. Calibration and perception (cluster, together)
Archive the old calibration first (on the Mac too, it has a copy; `sync.sh pull` never deletes):
```bash
mv experiment_results/appendix_calibrate_pixel_noise experiment_results/_archive_2026-10-14/
cluster/submit.sh appendix_calibrate_pixel_noise 3 06:00:00 1 4G
cluster/submit.sh ch6_perception 50 48:00:00 4 16G
```
- Calibration: the test episode is now 67 instead of 800 decisions, so each bisection step is ≈ 12× cheaper than
  in §2 (≈ 10 min per level instead of 1.4–2 h). 6 h leaves room for a `parametric` channel that again runs all
  60 steps.
- `ch6_perception`: new folder, no manifest patch. 50 entries (5 solvers × 10 seeds, `oracle_coarse` first,
  `ppo_pixel_fc` at 10–19), one per task. The probes now run 27 instead of 800 decisions, which made up most of the
  old 14–18 h per run, so 48 h is generous. `ppo_pixel_fc` trains with the borrowed rate (see A3).

#### B3. Paste the noise levels (Mac)
```bash
code/cluster/sync.sh pull appendix_calibrate_pixel_noise
python -m experiments.figures.appendix_calibrate_pixel_noise_figures
```
Replace `control`, `parametric` and `disturbance` in `CH6_NOISE_STD` (`measurement` stays hand-set). Check that
every printed ratio is close to its target (`NOISE_COST_TARGETS`); see the `parametric` note in §5.2. Commit, push,
`git pull` on the cluster. `ch6_noise` refuses to start until all four channels are filled.

#### A1. Learning-rate sweep (cluster, running)
Submitted as job 6538190 after
```bash
python cluster/patch_manifest.py appendix_rl_learning_rate solvers timesteps --write
cluster/submit.sh appendix_rl_learning_rate 45 72:00:00 2 8G
```
The grid has 270 entries, the new ones at 225–269, one per task. Wait for tasks 30–44 (*large*). If any times out,
resubmit the same `submit.sh` line; finished runs are skipped.

#### A2. Paste the learning rate (Mac)
```bash
code/cluster/sync.sh pull appendix_rl_learning_rate
python -m experiments.figures.appendix_rl_learning_rate_figures
```
Paste **only** the printed `ppo_fc_extractor` values (`small`, `medium`, `large`) into `SIZE_HYPERPARAMS` in
`settings.py`, next to its `pixel` entry. Leave the other solvers' values as they are. Until this is pasted,
`ch5_inflation_sizes` stops on the new solver with a "Run: ..." error.

#### A3. Check the borrowed *pixel* rate (Mac)
The other solvers' *pixel* rate is their *large* rate × √(200 / 18 432) ≈ *large* × 0.104.
- *large* = `6.7e-06` (as for `ppo_bottleneck` and `ppo_mlp`): the rule gives `7e-07`, the borrowed value. Nothing
  to rerun; drop the "TODO: placeholder" comment in `settings.py`.
- Anything else: two experiments used the borrowed rate, `ch6_solver_comparison` (`ppo_fc_extractor`, 70–120 h
  per run) and `ch6_perception` (`ppo_pixel_fc`). Either state in the thesis that the rate is borrowed from the
  identical trunk, or set the extrapolated value and rerun only those runs, once each experiment has finished:
  ```bash
  # cluster, after git pull of the new value
  mv experiment_results/ch6_solver_comparison/ch6_solver_comparison__ppo_fc_extractor__* experiment_results/_archive_<date>/
  python cluster/patch_manifest.py ch6_solver_comparison size_hyperparams --write
  cluster/submit.sh ch6_solver_comparison 50 6-00:00:00 2 16G
  mv experiment_results/ch6_perception/ch6_perception__ppo_pixel_fc__* experiment_results/_archive_<date>/
  python cluster/patch_manifest.py ch6_perception learning_rates --write
  cluster/submit.sh ch6_perception 50 48:00:00 4 16G
  ```
  Finished runs of the other solvers are skipped. Do the same on the Mac copy if it has those folders. This is a
  decision for the thesis, so settle it before submitting.

Commit, push, `git pull` on the cluster.

#### A4. The inflation runs, plus optional convergence (cluster)
```bash
python cluster/patch_manifest.py ch5_inflation_sizes solvers size_hyperparams hyperparams rl_timesteps          # dry run
python cluster/patch_manifest.py ch5_inflation_sizes solvers size_hyperparams hyperparams rl_timesteps --write
jid5=$(cluster/submit.sh ch5_inflation_sizes 30 72:00:00 2 8G --parsable)
```
Expected dry-run changes: `rl_timesteps[pixel]` 30 M → 10 M, `solvers` gains `ppo_fc_extractor`,
`size_hyperparams[ppo_fc_extractor]` and `hyperparams[ppo_fc_extractor]` appear, and the other solvers gain their
`pixel` learning rate (added after the run, and unused at these sizes). `patch_manifest.py` rewrites the whole
`size_hyperparams` entry, so if the dry run lists any **other** change for a solver, check that the stored runs were
trained with the new value before writing. The grid has 210 entries, new at 180–209, so 30 shards give one new run
each.

Optional, so the appendix shows the new solver's plateau as well (9 new runs, at 45–53):
```bash
python cluster/patch_manifest.py appendix_rl_convergence solvers size_hyperparams timesteps          # dry run
python cluster/patch_manifest.py appendix_rl_convergence solvers size_hyperparams timesteps --write
cluster/submit.sh appendix_rl_convergence 9 72:00:00 2 8G
```
Expected: only `ppo_fc_extractor`, the unused `pixel` learning rates and `timesteps[pixel]`. (The Mac's old copy,
with `ppo_encoder_scalable` at *small* on `1e-05`, was replaced by the cluster's run on 2026-10-14; the old one is in
`experiment_results/_archive_2026-10-14/`.)

#### A5. Chapter 5 noise (cluster)
```bash
cluster/submit.sh ch5_noise 15 01:00:00 2 8G --dependency=afterok:$jid5
```
No manifest patch needed: `ch5_noise` stores only the noise settings. Finished cells are skipped, so only the new
solver is scored.

#### B4. Chapter 6 noise (cluster)
Once B3 is pulled, `ch6_perception` has finished and A3 is settled (including any `ppo_pixel_fc` rerun):
```bash
cluster/submit.sh ch6_noise 40 02:00:00 2 8G
```
New folder, no patch (the old one is archived). The test episode is 67 instead of 800 decisions, so the 2 h that
were too short for 14 of 40 tasks in job 6476055 are now ample.

#### C1. Recount the policy size of older runs (cluster)
Run once every experiment holding PPO runs has finished (idempotent, so rerunning is harmless). Source experiments
come before their noise experiments, because a noise run copies the count of its source run. Runs made after
`84f5435` already store both counts, so this only matters for the older chapter 5 runs and the appendix sweeps:
```bash
srun --partition=smp --time=01:00:00 --mem=16G \
  python cluster/recount_policy_size.py ch5_inflation_sizes ch5_noise appendix_rl_learning_rate \
    appendix_rl_convergence ch6_solver_comparison ch6_perception ch6_noise           # dry run
# same command with --write
```
Do this on the cluster, not on the Mac: `sync.sh pull` overwrites a `record.json` recounted on the Mac with the
cluster's older version. (If you do recount on the Mac, rerun it after every pull.)

#### C1b. Count runaway test episodes of older runs
Since 2026-10-17 a nonlinear controller also counts as unstable if any clean test episode ends with the position
outside the initial range $[-10, 10]$ (`runaway_episodes`, `experiments/evaluate.py`; filtered in `tables.usable`).
Linear gains keep the spectral-radius rule and store `None`. New runs store the count. Older runs get it by replaying their clean test protocol (checked against the stored `test_costs`); the same
order applies, source experiments before their noise experiments:
```bash
python cluster/recount_runaway.py ch4_true_model ch4_identified_model ch5_inflation_sizes \
  appendix_rl_convergence appendix_rl_learning_rate ch4_noise ch4_true_model_noise ch5_noise --write
```
This was run on the Mac on 2026-10-17, so the same caveat holds: rerun it after every `sync.sh pull` of these
experiments, or run it once on the cluster as well. The pixel chapter keeps its own `terminated_rate` instead.

#### C2. Figures (Mac)
```bash
code/cluster/sync.sh pull        # or each experiment by name
python -m experiments.figures.appendix_rl_learning_rate_figures
python -m experiments.figures.appendix_rl_convergence_figures
python -m experiments.figures.ch5_inflation_figures
python -m experiments.figures.ch5_noise_figures
python -m experiments.figures.ch6_solver_comparison_figures
python -m experiments.figures.appendix_calibrate_pixel_noise_figures
python -m experiments.figures.ch6_perception_figures
python -m experiments.figures.ch6_noise_figures
```
Then rebuild the thesis. The text that depends on the counts: §4.2 complexity paragraph, §4.4 discussion
("half the parameters"), `def:policy-size` (frozen parameters count), and the policy-size discussion of chapter 5.

### 5.5 `ch6_solver_comparison`: rerun at 10 M steps (running, job 6538235)
At 30 M steps this experiment could not finish: the `ppo_mlp` runs took 46.6 h each on the cluster (job 6476056,
167 604 s for 30 M, so ≈ 1.55 h per 1 M steps, 1.3–2.2× the Mac time of §2), and `ppo_bottleneck` /
`ppo_encoder_scalable` were still running at the 72 h limit. `RL_MAX_TIMESTEPS["pixel"]` is therefore 10 M.
Cluster wall time per run at 10 M (`ppo_fc_extractor` has the same trunk as `ppo_bottleneck`, 9.67 M vs. 9.66 M
parameters at *pixel*):

| Solver | Per run at 10 M |
|---|---|
| `baseline` | ≈ 1.2 h (job 6538235; the test protocol at *pixel*) |
| `ppo_mlp` | 10–17 h (job 6538235) |
| `ppo_encoder_scalable` | ≈ 20–50 h (estimate) |
| `ppo_bottleneck`, `ppo_fc_extractor` | ≈ 70–120 h (estimate) |

The learning rate anneals to zero at the cap (§2), so a 10 M run is **not** the first third of a 30 M run: every
PPO run was retrained, and the 30 M runs (in `_archive_2026-10-14/` on the cluster) must not be mixed in. The
`pixel` learning rates stay as they are (they scale with the input dimension, not the budget). The grid has 50
entries, solver by solver, ten seeds each, `ppo_fc_extractor` last at 40–49, submitted as
```bash
cluster/submit.sh ch6_solver_comparison 50 6-00:00:00 2 16G
```
When the remaining tasks finish, update the table above from `sacct -j 6538235 --format=JobID,State,Elapsed`.

**For the thesis:** chapter 5's solver comparison now trains for 10 M steps at *pixel*, a third of the *large*
budget (30 M). Say so where the budget is introduced, and that the cut is for compute reasons.
