## Stable Baselines
https://stable-baselines3.readthedocs.io/en/master/
https://github.com/DLR-RM/stable-baselines3

## Gymnasium
https://gymnasium.farama.org/environments/classic_control/

## Papers to read for low dimensional versions
Not so relevant: https://proceedings.neurips.cc/paper_files/paper/2025/hash/68c5d2f918adf5638071fd4191c91145-Abstract-Conference.html
Not so relevant: https://proceedings.neurips.cc/paper_files/paper/2024/hash/371713c3e5314dff9483c62c5abb98a8-Abstract-Conference.html

### Relevant for robotics: https://ieeexplore.ieee.org/abstract/document/9635936
- Source for position-velocity encoder: https://arxiv.org/html/1705.09805v3
- AutoEncoder
- using one nn for state estimation, then policy network

### Robotic priors
- Original paper: Jonschkowski & Brock, Learning State Representations with
  Robotic Priors, Autonomous Robots 2015. Introduces five priors: simplicity, temporal coherence,
  proportionality, causality, repeatability. Simplicity is imposed by
  fixing a low dimensionality.
- Using Botteghi et al., https://arxiv.org/pdf/2107.01667 for continuous action spaces

### Orthogonal bottlenecks: https://arxiv.org/abs/2605.26012
- The projection B is sampled once by QR of a Gaussian and fixed
- k=2 is sufficient for baseline performance on classic control
- In deep linear networks, Saxe et al. (2014)
and Hu et al. (2019) show that orthogonal initialization can provably accelerate gradient descent
compared to Gaussian initialization
- suggests certain standards for testing https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

### Position Velocity Encoders: https://arxiv.org/html/1705.09805v3
- using pixel inputs to estimate position and velocity


Possible approach for RL:
- Comparison with LQR
    - use linear PPO, fully connected PPO, orthogonal bottleneck PPO and Low-dimensional Encoder + fully connected PPO
- Extension to Pixel Input
    - fully connected, bottleneck, low-dimensional encoder, PVE