- cite "Underactuated Robotics" for most of the control theory in preliminaries
- mohamadi et al for sample complexity in model-free?
- Dreamer: Action in low-latent space, mapping from high dimensional observation
- Kumar: implicit under parameterization inhibits efficient learning
- Rozada: Matrix Low-Rank Trust Region Optimization - maybe interesting for chap5?
- Igl: Bottleneck appraoch
- Li: Intrinsic Dimensions, also interesting as a low-dimensional approach


## Text explanations
- explain & understand deeply how we recover cost matrices via N4SID
- explain why CE_DARE needs a collapsed & simplified identified model
- reason why N4SID is not scalable to pixel size
- explain how we score how we compare the latent vector to pos-vel via ridge regression
- attribute ridge regresseion against ground truth state in code
- pixel_solver in sovlers has "timesteps", which refers to agent steps, overriden by setting.RL_steps, but is not the same env steps as in chap5
- justify dimension 2 for the different ppo models (ortho an est)


## Code
- check ppo_linear, mlp and N4SID encoder
- check pixel_fc, pixel_posvel
- rework noise plots
- where is the bottleneck approach?

- we don't take the best policy, but the last, should be improved - is there some way we can evaluate what the best policy is without looking at the model?
- drop "Total test cost" from tables and plots
- calibrate noise levels based on how much they increase the base cost: x1.2, x1.5, x3, x10, better than based on visual on trajectories
- disturbance is applied to all states equally, which doesn't make too much sense - should be a physical push to the position, better this way
- run noise comparisons at large inflation for ch4 & ch5, now possible thanks to the server
- include real "convergence" criteria for each solver, based on relative improvement, together with environment steps / iterations count as a reported number
- also do this for the RL models, as they don't seem to converge yet
- check the lambda settings for the model-based spi, does it make sense to handle them differently than standard SPI?
- extend the appendix spi_mf sweep across the inflation sizes
- rethink whether it makes sense to use exactly the same "reward" approach between chap4 and chap5
- action_clips counter is broken, and generally not the same between RL and LQR
- why do we use the 800 step rollout for ppo, vs. the 2000 step for eval (discounted / undiscounted)
- need to normalize the observations for the pixel size, or generally, as they are shrunken by the lifting