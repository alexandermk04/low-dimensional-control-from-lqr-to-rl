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
- document functions, etc.
- write preliminaries