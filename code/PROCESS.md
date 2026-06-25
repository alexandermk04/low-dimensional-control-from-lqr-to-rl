## Finding fitting parameters for estimation
- to little rollout underestimates cost


## Fixing divergence in inflated input dimensions
### MF Natural Gradient
- using the pseudoinverse for sigma with a cutoff makes the estimation become stable

### SPI
- Orthonormal lifts on the inflation matrices instead of completely random
- Implement rollout based linesearch
- Divergence protection not good enough
- Initialize with values such that lambda * stepsize is aroun 0.1 - 0.5
- get the same rng state for both cost evaluations
- two-point estimator

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