import numpy as np
import matplotlib.pyplot as plt

from basetypes import SimulationResult


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

def visualize_multiple_results(results: dict[str, SimulationResult], metadata: dict[str, str], save_to: str = None, dt=1):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.title('Position')
    for label, result in results.items():
        time_sec = np.arange(result.x_traj.shape[0]) * dt
        plt.plot(time_sec, result.x_traj[:, 0], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('Velocity')
    for label, result in results.items():
        time_sec = np.arange(result.x_traj.shape[0]) * dt
        plt.plot(time_sec, result.x_traj[:, 1], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Control')
    for label, result in results.items():
        time_sec = np.arange(result.u_traj.shape[0]) * dt
        plt.plot(time_sec, result.u_traj[:, 0], label=f'{label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Control')
    plt.legend()

    # For adding run metadata
    plt.subplot(2, 2, 4)
    plt.axis('off')

    text_content = ""
    for key, value in metadata.items():
            text_content += f"{key}: {value}\n"
    plt.text(0.5, 0.5, text_content, 
                 fontsize=12, 
                 horizontalalignment='center', 
                 verticalalignment='center'
                 )

    plt.tight_layout()

    if save_to is not None:
        plt.savefig(f"../../Visualizations/{save_to}.png")
    else:
        plt.show()