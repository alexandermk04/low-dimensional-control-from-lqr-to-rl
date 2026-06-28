import numpy as np
import gymnasium as gym
import pygame

from gymnasium import spaces

class DoubleIntegratorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self, 
            x_threshold: float,
            v_threshold: float,
            f_threshold: float,
            max_steps: int = 200,
            goal_tolerance: float = 0.05,
            dt: float = 0.01,
            render_mode: str | None = None
        ):
        
        super().__init__()

        self.x_threshold = x_threshold
        self.v_threshold = v_threshold
        self.f_threshold = f_threshold
        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        self.dt = dt

        self.render_mode = render_mode

        # Action space, force
        act = np.array([self.f_threshold], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-act,
            high=act,
            dtype=np.float32
        )

        obs = np.array([self.x_threshold, self.v_threshold], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs,
            high=obs,
            dtype=np.float32
        )

        # Reset all internal state variables
        self.state: np.ndarray | None = None
        self.steps = 0

        # For rendering
        self.screen = None
        self.clock = None
        self.screen_dim = 500

    def get_random_start_state(self):

        x_bound = self.x_threshold
        v_bound = self.v_threshold / 10 # SHOW: Start with a small velocity

        x = np.random.uniform(-x_bound, x_bound)
        v = np.random.uniform(-v_bound, v_bound)

        return np.array([x, v], dtype=np.float32)
    
    def calculate_reward(self, x, v, force):

        # TODO: Find a way to translate the reward function from LQR to RL. Currently just a simple quadratic cost function is used.

        reward = -(x ** 2 + 0.1 * v ** 2 + 0.01 * force ** 2)

        return reward

    def _get_obs(self):
        return self.state
    
    def _get_info(self) -> dict:
        x, v = self.state
        return {"position": x, "velocity": v, "steps": self.steps}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.state = self.get_random_start_state()
        self.steps = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")
        
        # Action should be an array of shape (1,)
        force = np.clip(action[0], -self.f_threshold, self.f_threshold)

        x, v = self.state

        # We assume a mass of 1 for simplicity, so acceleration = force
        x = x + v * self.dt
        v = v + force * self.dt

        # Enforce the state constraints
        v = np.clip(v, -self.v_threshold, self.v_threshold)

        reward = self.calculate_reward(x, v, force)

        terminated = False

        goal = np.abs(x) < self.goal_tolerance and np.abs(v) < self.goal_tolerance
        if goal:
            reward += 10.0  # Bonus for reaching the goal
            terminated = True

        # Make the cart hit a wall and penalize it
        if np.abs(x) > self.x_threshold:
            x = np.clip(x, -self.x_threshold, self.x_threshold)
            v = 0.0
            reward -= 15.0  # Penalty for going out of bounds

        # Update the state
        self.state = np.array([x, v], dtype=np.float32)
        self.steps += 1

        truncated = self.steps >= self.max_steps

        info = self._get_info()
        info["is_success"] = goal

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode is None:
            return
        
        width, height = self.screen_dim, self.screen_dim // 2

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Double Integrator Environment")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Initialize the background
        field = pygame.Surface((width, height))
        field.fill((255, 255, 255))

        line_y = height // 2

        # The line representing the ground
        pygame.draw.line(field, (100, 100, 100), (0, line_y), (width, line_y), 2)

        # Goal point
        pygame.draw.circle(field, (0, 150, 0), (x_to_pixel(0, self.x_threshold, width), line_y), 5)

        x, v = self.state

        # Our cart
        cart_x = x_to_pixel(x, self.x_threshold, width)
        pygame.draw.rect(field, (30, 90, 200), (cart_x - 10, line_y - 10, 20, 20))

        # Arrow for velocity, scaled for visibility
        pygame.draw.line(field, (200, 40, 40), (cart_x, line_y), (int(cart_x + v * 20), line_y), 2)

        self.screen.blit(field, (0, 0))

        # Update the display and tick the clock with 30 FPS
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

# Helper function for rendering
def x_to_pixel(x, x_threshold, screen_width):
    return int((x + x_threshold) / (2 * x_threshold) * screen_width)


if __name__ == "__main__":
    # Sanity check for the environment
    from stable_baselines3.common.env_checker import check_env

    env = DoubleIntegratorEnv(
        x_threshold=2.0,
        v_threshold=5.0,
        f_threshold=2.0,
    )

    check_env(env)