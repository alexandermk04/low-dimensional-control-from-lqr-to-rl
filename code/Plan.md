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

=> QUESTION: Where do I apply the noise? Only when simulating in the end, or already inside the sampling as well?