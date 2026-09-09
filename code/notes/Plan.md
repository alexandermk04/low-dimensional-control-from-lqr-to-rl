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

## Chapter 4
Comparison of Baseline Solution with Model-based, Model-based + System Identification, Model-free Policy Gradient Solvers - first noise free comparison using plots 1-3, 6, and table 1, 2 (but without policy complexity here), then tables 3,4 and and plots 4,5. So we compare the base controllers performance under normal behavior, as well as noise application. Inflation is also introduced here. Overall: Basic solvers, set data sample sizes based on experiments with different inflation intensities (base (2 in, 1 out), small (10 in, 5 out), medium (50 in, 25 out), large (200 in, 100 out)).

## Chapter 5
First introduction of model-based spi, explanation why model-free is excluded. Experiments with model-based spi + model-estimator; plots 1-3, 6 and tables 2,3 (here we use sample sizes selected based on results from 4) for comparison only with mb + model-estimator between different inflation sizes.

Introduce linear-ppo, fully-connected ppo and state-encoder, do same comparisons with different inflation sizes for the state-encoder only.

Then afterwards same plots and tables for comparison between all those newly introduced solvers (with only medium sized inflation), as well as the baseline and mb + model-estimator from chapter 4 in the inflated dimensions, include additionally policy complexity here.

Afterwards a section comparing only baseline with model-based + estimator of policy gradient and spi and fully-connected ppo and fully connected + state encoder under the different noises, like we did in chapter four, but here with medium inflation.

## Chapter 6
Introduce how we now observe our environment via a camera, explain which assumptions we take. Introduce the Position Velocity Encoder shortly, mainly link the real paper. Only do comparison of performance with plots 1-3, 6, maybe 7 if shows difference, as well as table 2 between baseline, state-encoder + fully connected ppo and pve. No noise comparison with be done here.

Add additional plots analyzing how well each solver captures velocity and position during testing - bar plots with accuracy per solver side by side.

## Plotting & Table goals
The goal would be to create the table and plots within latex, to have a uniform styling of my paper - whatever the best approach would be for that.