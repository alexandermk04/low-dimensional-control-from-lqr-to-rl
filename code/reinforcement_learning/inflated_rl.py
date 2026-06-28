# Example from: https://github.com/DLR-RM/stable-baselines3

import gymnasium as gym

from stable_baselines3 import PPO

from reinforcement_learning.double_integrator_environment import DoubleIntegratorEnv

if __name__ == "__main__":
    env = DoubleIntegratorEnv(
        x_threshold=2.0,
        v_threshold=5.0,
        f_threshold=10.0,
        render_mode="human",
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000, progress_bar=True)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render(mode="human")
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()