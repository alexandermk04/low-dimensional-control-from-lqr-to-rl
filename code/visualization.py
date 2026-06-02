import numpy as np
import matplotlib.pyplot as plt

from basetypes import SimulationResult, SimulationResultPlotType


def visualize_result(result: SimulationResult, dt=1):
    time_sec = np.arange(result.x_traj.shape[0]) * dt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_sec, result.x_traj[:, 0], label='Position')
    plt.plot(time_sec, result.x_traj[:, 1], label='Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time_sec, result.u_traj[:, 0], label='Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Control')
    plt.legend()

    plt.show()

def visualize_result_single(result: SimulationResult, type: SimulationResultPlotType, dt=1):
    time_sec = np.arange(result.x_traj.shape[0]) * dt

    y_label = {
        SimulationResultPlotType.POSITION: "Position",
        SimulationResultPlotType.VELOCITY: "Velocity",
        SimulationResultPlotType.ACCELERATION: "Acceleration"
    }[type]

    data = {
        SimulationResultPlotType.POSITION: result.x_traj[:, 0],
        SimulationResultPlotType.VELOCITY: result.x_traj[:, 1],
        SimulationResultPlotType.ACCELERATION: result.u_traj[:, 0]
    }[type]

    plt.figure(figsize=(10, 6))
    plt.plot(time_sec, data)
    plt.xlabel('Time (s)')
    plt.ylabel(y_label)

    plt.show()