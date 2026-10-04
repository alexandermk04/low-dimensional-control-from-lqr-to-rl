- cite "Underactuated Robotics" for most of the control theory in preliminaries
- mohamadi et al for sample complexity in model-free?
- Dreamer: Action in low-latent space, mapping from high dimensional observation
- Kumar: implicit under parameterization inhibits efficient learning
- Rozada: Matrix Low-Rank Trust Region Optimization - maybe interesting for chap5?
- Igl: Bottleneck appraoch
- Li: Intrinsic Dimensions, also interesting as a low-dimensional approach

## Sources for fixing RL approaches
- Huang et al. 2022, "The 37 Implementation Details of Proximal Policy Optimization", ICLR Blog Track: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Stable-Baselines3 docs, "Reinforcement Learning Tips and Tricks" (VecNormalize, action rescaling): https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Stable-Baselines3 docs, "Custom Policy Network" (`share_features_extractor`): https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html


## Text explanations
- explain why CE_DARE needs a collapsed & simplified identified model
- reason why N4SID is not scalable to pixel size
- explain how we score how we compare the latent vector to pos-vel via ridge regression
- attribute ridge regresseion against ground truth state in code
- pixel_solver in sovlers has "timesteps", which refers to agent steps, overriden by setting.RL_steps, but is not the same env steps as in chap5
- justify dimension 2 for the different ppo models (ortho and est)
- explanation for how stopping works for the different solvers
- explain the RL observation scaling s = sqrt(p/n), and why actions are not scaled
- RL action clipping vs. LQR without it
- check years for sources


## Code
- check ppo_linear, mlp and N4SID encoder
- check pixel_fc, pixel_posvel
- rework noise plots
- remove test costs from everywhere, recheck "convergence" stopper
- ch4 identified model needs base curves iterations plots instead of training suboptimality

- create visualization of the setup for pixel
- is the identification for Q and R the same as in 5.4 of N4SID?

- Do we need to calibrate Control Noise for each size?

## Questions
- Should I introduce noise also to the pixelated / pixel size approach?


## Agenda 2.10
Structural questions:
- general look at the structure
- test setup, convergence, noise injection, conventions: Stay in Preliminaries (Chap 2) or move to Approach (Chap 3.2)?
=> combine chapter 2 and 3

- comparison of RL solvers under "pixel size" (18432 in, 256 out) in RL approaches (Chap 5) or Pixelated Environment (Chap 6)?
=> move to chapter 6

Chapter 4, True Model:
- explain that for the application of DARE, we need to reduce the dimension of the model "artificially", otherwise matrix is singular
    => is that a problem, or ok? The controller would then act in this "lower" dimension, hence is automatically smaller
-> write email

- with the trained controllers then evaluation under noise? -> noise

Chapter 6, Pixelated / Perception Environment:
- right now, we just compare different solvers on this new environment, however don't really test the robustness to changes
    => should we introduce some noise additionally?

-> experiment with noise

Timeline:
- right now the major blocker for results is the selection of RL parameters like learning rate and convergence
    => should finish over the weekend, can then send results
- still, I can already write down the first draft of the experiment setups, so that I only need to fill in the results
- chapter 1,2,3 are finished in their first version, chapter 4 the experiment setup mostly written
- goal is to finish by the 23rd, then send it to you for any open points
