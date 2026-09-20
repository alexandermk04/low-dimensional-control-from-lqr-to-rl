- cite "Underactuated Robotics" for most of the control theory in preliminaries
- mohamadi et al for sample complexity in model-free?
- Dreamer: Action in low-latent space, mapping from high dimensional observation
- Kumar: implicit under parameterization inhibits efficient learning
- Rozada: Matrix Low-Rank Trust Region Optimization - maybe interesting for chap5?
- Igl: Bottleneck appraoch
- Li: Intrinsic Dimensions, also interesting as a low-dimensional approach

- explain & understand deeply how we recover cost matrices via N4SID
- explain why CE_DARE needs a collapsed & simplified identified model
- reason why N4SID is not scalable to pixel size
- explain how we score how we compare the latent vector to pos-vel via ridge regression
- pixel_solver in sovlers has "timesteps", which refers to agent steps, overriden by setting.RL_steps, but is not the same env steps as in chap5


- add policy complexity to chap4 identified model
- explore suboptimality as a log y-axis