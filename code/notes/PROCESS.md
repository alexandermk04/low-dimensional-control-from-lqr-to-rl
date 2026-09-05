## Finding fitting parameters for estimation
- to little rollout underestimates cost


## Fixing divergence in inflated input dimensions
### MF Natural Gradient
- using the pseudoinverse for sigma with a cutoff makes the estimation become stable

### SPI
- Orthonormal lifts on the inflation matrices instead of completely random
- Implement rollout based linesearch, better with fixed LR
- two-point estimator is really important!!!

## Testing performance of Full-rank vs. low-rank
- Started with optimal policy, full rank better
- Moved to previously tested -> now SPI much better
- Increased LR for normal, now quite similar, but diverges from time to time
- Realized that K_0 is still one-rank, implementation for full-rank with stable initial

- fix normal SPI
- start close to optimum
-...

## Findings regarding divergence of SPI
- paper mentions system of (3, 3) dimensions with lambda = 3000
diverges for steps larger than 3.7e-4
- 3.7e−5, 3.7e−6 as stepsize converges to optimal cardinality
- 3.7e−7, 3.7e−8, 3.7e− 9 when used for stepsize didn't converge at all
- largest fixed stepsize for stable systems:
    - lambda 10^3; step little below 10-4
    - lambda 10^4; step between 10-5 and 10-6
    - lambda 10^5; step close to 10-7
- try laplacian systems with (3, 3), (20, 20), (10^3, 10^3)
- start with stepsize 1/lambda, beta 1/2, epsilon 10-6
- evaluate based on cardinality

## 3rd of August:
- Extending the base clases to using the settings parameters
- Random initialization of initial state, instead of fixed
- SPI behaves weirdly for random initial states

## 4th of August:
- Check whether policy gradients work with random initial position
- If not, we revert and just use a fixed starting point

## 5th of August:
- Reset initial state of LQR before each iteration in the solvers
-> Is already reset inside the estimate_gradient function before each trajectory run
-> model based also now included
=> still doesn't seem to converge in the gradient only versions, needs revision of literature

## 7th of August:
- Include setting for random sample vs. stable initial state
-> Good seeds with success: 2022
-> Mentions in the paper that model free approach is hard to implement
- Open TODO: Fix cost estimation throughout - currently uses sometimes finite horizon estimates, should use infinite, model based calculations
- TODO: fix model based SPI for inflated settings
- TODO: Fix colors in cost plots
- TODO: Investigate convergence behavior with certain seeds
- TODO: Fix noise implementation for model-based solvers
- TODO: Check, whether diabling noise on reset of state is wanted behavior
- TODO: Integrate ZOH integration for gymnasium environment
- document functions, etc.
- write preliminaries

## 21st of August
- added tqdm to watch the progress of true cost evaluation during iterations

## 23rd of August
- added seeding to improve reproducibility and comparability
- TODO: Need to reset seeding before each simulation run, to get exactly the same results here (for noise)
- !!Finding: SPI needs only a tiny fraction of the iterations to reduce the matrix to low rank and lower the costs - overiterating makes it worse, until divergence

## 26th of August
- check whether notation for noise sampling is correct of dimension

# 30th of August
- TODOs: 
  - Readjust learning rate parameters
  - Make evaluation LQR system identical for fair simulation comparison (in relation to initial state)

## 2nd of September: unifying the LQR and RL halves
- potentially removing gamma from metrics if not needed

## 3rd of September
- read up on possible RL architectures
- investigated possible parameter combinations for converging SPI
  - would require large m (number of trajectories) to be able to reduce rank without diverging (around 120M env steps)
  - needs decision: should we allow this type of budget? does this contradict our assumption?

## Agenda 4th of September
Important decisions:
- SPI:
  - Model-based works well, tells the story I want
  - Model-free doesn't converge at all, or not low-rank
  - Can converge, however only with 20000+ trajectories
  - Question: Leave this out as unsuccessful, report as such, details in appendix? 
  -> focus more on mb, maybe model-sampling first, then use with mb
  -> known difficulty of model-free approaches

- (Robustness) Metrics / Experiment setup
  - Run with 10 different seeds
  - True Cost
  - Supoptimality: (J - J*) / J*
  - Policy Complexity: No of params -> reason why
  - Environment Steps used -> Data Size, learning time (rollout, etc.; sample complexity)
  - Convergence
  - How stable is a policy if the dynamics change? -> different environments, post-training, mass, friction change (dynamics disturbances)
  - Ok like this, any other metric?

- Settings for Noise / Algorithms:
  - How do I decide number of trajectories, etc.?
  - How large should the noise be? - are 3 different scales that I compare enough?

- Comparison with RL chapter 5:
  - Linear PPO, allowing to extract gain matrix
  - Fully connected PPO
  - Orthogonal-bottleneck PPO -> maybe exclude regarding time
  - State Encoder + PPO
  - Question: Ok, preferences, leave some out?
- Chapter 6, perception based
  - Can this be shortened / cut if time runs out?
  - Introduce Position-Velocity Encoder as well

- Access to high compute renewable?