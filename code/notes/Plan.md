## Order

### LQR Part
- Baseline Calculation
- Model-based / Model-free gradient
- Low-Rank Structured Policy Iteration
- TODO: Model-free Structured Policy
- TODO: Data-Enabled PO
- TODO: Geometric...

## Robustness Evaluations
- Parametric uncertainty -> slightly modified A and B matrices during runtime
    - time-invariant
    - time-variant uncertainty (modify during simulation)
- Multiplicative control noise -> randomly modify policy via scaled error
    - matrix multiplication for vector
- Measurement noise -> randomly add noise to state observation
- External disturbances -> disturbance vector to state transitions

## Artificial Inflation
- Cameras? -> need to use CNN like in paper, is that wanted?
- Matrix multiplication -> really artificial creation

## RL
- Modify stable baselines
- Full-Rank / Low-Rank implementation
- Comparison with different Robustness levels

- compute costs exactly for a given K instead of estimates
- natural gradient, adding epsilon times identity before inverting it in case of problems (1e-3)
- gradients start visually with different K_0s, investigate -> why does model based explode?
- fill in registration form

## Final setting decisions:
### Experiment setups:
- everything averaged over 10 different seeds
- noise only applied during testing, not training
- 100 test runs per seed
- run them for the same length as during training
- Noise run in 5 different scales (0, little effect, standard effect, strong effect, divergence) -> will be fixed after a little bit of trial
- Each noise channel isolated run, then all together

### Metrics
- True Cost after training
- Suboptimality with J*
- Policy Complexity
- Sample Size for training (fixed rollout lengths of 200, 500, 800) with m = 100, 200, 500
- iterations fixed at 500 (unless to short), check convergence possibility!!
- Test evaluation: Model run for a fixed number of steps (2000) and m = 100, then
accumulated cost averaged over runs and seeds

### Plots
1. Box plots for True Cost / Suboptimality
2. Line Graphs with mean and range for training process
3. Simulation (position, velocity, acceleration) graphic showing different trained controllers for the same starting state
4. Simulation showing one specific controller with the same starting state for the different noise levels applied
5. Simulation showing one specific controller with the same starting state for different noise channels at the same level applied
6. Box Plots for Cost accumulated during tests
7. Barplots for policy complexity

### Tables
1. Table with y-axis metrics listed (true cost post-train, subopt, test-average cost), on the x-axis one policy with the different sample size combinations (should be 9 columns)
2. Similar table 1., but instead of combinations, different policy solvers on x-axis (mf_npg, etc.) and on y-axis add policy complexity
3. Table as in 2., but instead of different solvers, one solver with one noise at different strengths applied
4. Table as in 3., but with different noise channels at one strength as columns


## Appendix
- Sweep l, m combinations to determine values for remaining experiments
- Sweep T, hankel combinations to determine values for remaining experiments
- Sweep different lambda, stepsize combination for spi_mf to determine divergence

## Chapter 4

Both run over all inflation sizes

### Track 1: true model, exact gradient
- baseline, mb_pg, mb_spi comparison (pure)
- TODO: comparison under noise on one inflation size

### Track 2: unknown model
- baseline, ce_dare, mb_est, spi_est, mf_npg, spi_mf (if possible)
- comparison under noise (one size)
- TODO: Comparison with suboptimal identification under noise (only one size)

## Chapter 5
- comparison across sizes of baseline, ppo_linear, ppo_mlp, ppo_encoder (N4SID), ppo_encoder_scalable, ppo_bottleneck
- comparison under noise (one size)
- comparison of selected from ch4 and ch5 under pixel size with noise
- ppo_encoder is unsupported at pixel size (N4SID cannot identify 18432 dimensions), replaced with ppo_encoder_scalable here

## Chapter 6
- comparison of oracle_coarse, ppo_pixel_fc, ppo_pixel_ortho, ppo_pixel_estimator,
  ppo_pixel_posvel
- r2_position, r2_velocity for how well they identify the real state
- show some example renderings to explain the architecture

## Plotting & Table goals
The goal would be to create the table and plots within latex, to have a uniform styling of my paper.