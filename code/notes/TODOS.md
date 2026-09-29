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
- explain & understand deeply how we recover cost matrices via N4SID
- explain why CE_DARE needs a collapsed & simplified identified model
- reason why N4SID is not scalable to pixel size
- explain how we score how we compare the latent vector to pos-vel via ridge regression
- attribute ridge regresseion against ground truth state in code
- pixel_solver in sovlers has "timesteps", which refers to agent steps, overriden by setting.RL_steps, but is not the same env steps as in chap5
- justify dimension 2 for the different ppo models (ortho and est)
- explanation for how stopping works for the different solvers
- explain the RL observation scaling s = sqrt(p/n), and why actions are not scaled
- RL action clipping vs. LQR without it


## Code
- check ppo_linear, mlp and N4SID encoder
- check pixel_fc, pixel_posvel
- rework noise plots
- where is the bottleneck approach?

- Do we need to calibrate Control Noise for each size?

## Questions
- Comparison on the true model - does this make sense?
- maybe move noise implementation, as well as metrics and "general test setup" into approach, together with dimensional lifts?