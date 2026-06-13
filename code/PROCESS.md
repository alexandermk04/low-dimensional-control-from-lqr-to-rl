## Finding fitting parameters for estimation
- to little rollout underestimates cost


## Fixing divergence in inflated input dimensions
### MF Natural Gradient
- using the pseudoinverse for sigma with a cutoff makes the estimation become stable

### SPI
- Orthonormal lifts on the inflation matrices instead of completely random

## Testing performance of Full-rank vs. low-rank
- Started with optimal policy, full rank better
- Moved to previously tested -> now SPI much better
