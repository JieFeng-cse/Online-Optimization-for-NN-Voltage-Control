import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np

    
def plot_trajectory_and_slopes(traj_1, traj_2, traj_3, 
                               title_1=r'Trajectory of $\mathbf{v}_t$',
                               title_2=r'Trajectory of $\mathbf{q}_t$',
                               title_3=r'Trajectory of $\mathbf{b}_{\theta_t^2}$'):
    
    traj_1 = np.array(traj_1)
    traj_2 = np.array(traj_2)
    traj_3 = np.array(traj_3)

    num_states = traj_1.shape[1]
    num_slopes = len(traj_2[0])

    time_steps = np.arange(len(traj_1))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0), dpi=400)

    # Plot Voltage Trajectories
    for i in range(num_states):
        axes[0].plot(np.arange(len(traj_1)), traj_1[:, i], linewidth=0.8, label=f'Bus {i + 1}')
    axes[0].set_xlabel('Time Steps', fontsize=6)
    axes[0].set_ylabel('Voltage [p.u.]', fontsize=6)
    axes[0].set_title(title_1, fontsize=7)
    axes[0].grid(True, linewidth=0.3)
    axes[0].tick_params(axis='both', labelsize=5)

    # Plot Reactive Power Trajectories
    for i in range(num_slopes):
        axes[1].plot(np.arange(len(traj_2)), traj_2[:, i], linewidth=0.8, label=f'Bus {i + 1}')
    axes[1].set_xlabel('Time Steps', fontsize=6)
    axes[1].set_ylabel('Reactive Power', fontsize=6)
    axes[1].set_title(title_2, fontsize=7)
    axes[1].grid(True, linewidth=0.3)
    axes[1].tick_params(axis='both', labelsize=5)

    # Plot Setpoint Trajectories
    for i in range(num_slopes):
        axes[2].plot(np.arange(len(traj_3)), traj_3[:, i], linewidth=0.8, label=f'Bus {i + 1}')
    axes[2].set_xlabel('Time Steps', fontsize=6)
    axes[2].set_ylabel('Setpoint', fontsize=6)
    axes[2].set_title(title_3, fontsize=7)
    axes[2].grid(True, linewidth=0.3)
    axes[2].tick_params(axis='both', labelsize=5)

    # Unified Legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=6, fontsize=5, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('figures/control_traj.png', bbox_inches='tight')
    plt.show()

def plot_v_and_q(traj_1, traj_2, 
                               title_1=r'Trajectory of $\mathbf{v}_t$',
                               title_2=r'Trajectory of $\mathbf{q}_t$'):

    traj_1 = np.array(traj_1)
    traj_2 = np.array(traj_2)

    num_states = traj_1.shape[1]
    num_slopes = traj_2.shape[1]

    time_steps = np.arange(len(traj_1))

    # Figure adjusted for one-column paper (~3.5 inches wide)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 1.8), dpi=400)

    # Plot Voltage Trajectories
    controlled_bus = [2,7,9]
    for i in range(num_states):
        ax1.plot(time_steps, traj_1[:, i], linewidth=0.8, label=f'Bus {i+1}')
    ax1.set_xlabel('Time Steps', fontsize=7)
    ax1.set_ylabel('Voltage [p.u.]', fontsize=7)
    ax1.set_title(title_1, fontsize=8)
    ax1.grid(True, linewidth=0.3)
    ax1.tick_params(axis='both', labelsize=6)
    ax1.legend(fontsize=4, loc='upper right', ncol=3, frameon=False)

    # Plot Reactive Power Trajectories
    for i in range(num_slopes):
        ax2.plot(np.arange(len(traj_2)), traj_2[:, i], linewidth=0.8, label=f'Bus {controlled_bus[i]}')
    ax2.set_xlabel('Time Steps', fontsize=7)
    ax2.set_ylabel('Reactive Power [MVar]', fontsize=7)
    ax2.set_title(title_2, fontsize=8)
    ax2.grid(True, linewidth=0.3)
    ax2.tick_params(axis='both', labelsize=6)
    ax2.legend(fontsize=5, loc='best', frameon=False)

    plt.tight_layout(pad=0.8)
    plt.savefig('figures/control_traj.png', bbox_inches='tight')
    plt.show()