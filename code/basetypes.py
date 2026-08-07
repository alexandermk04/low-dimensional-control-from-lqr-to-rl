from dataclasses import dataclass
import numpy as np

SEED = 2024

@dataclass
class StateSpaceModel:
    A: np.ndarray
    B: np.ndarray

@dataclass
class StepResult:
    y_t: np.ndarray
    v_t: np.ndarray
    x_t: np.ndarray
    u_t: np.ndarray
    cost: float

@dataclass
class SimulationResult:
    x_traj: np.ndarray
    u_traj: np.ndarray