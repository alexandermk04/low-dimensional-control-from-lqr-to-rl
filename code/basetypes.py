from dataclasses import dataclass
import numpy as np

@dataclass
class StateSpaceModel:
    A: np.ndarray
    B: np.ndarray
    x_0: np.ndarray

@dataclass
class SimulationResult:
    x_traj: np.ndarray
    u_traj: np.ndarray

@dataclass
class SimulationResultPlotType:
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"