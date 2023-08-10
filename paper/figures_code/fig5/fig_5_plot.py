import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from utils import plot_maze

load_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig5'))

def main():

    with open(os.path.join(load_path, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(2,3,1)
    plot_maze(ax1, np.load(os.path.join(load_path, 'q_mb.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(2,3,2)
    q_explore_replay_diff = np.load(os.path.join(load_path, 'q_explore_replay_diff.npy'))
    q_explore_replay_diff[q_explore_replay_diff <= agent.xi] = np.nan
    plot_maze(ax2, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax2.set_title(r'Exploratory replay', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(2,3,3)
    plot_maze(ax3, np.load(os.path.join(load_path, 'q_explore_replay.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax3.set_title(r'Updated exploratory policy', fontsize=16)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax5 = fig.add_subplot(2,3,4)
    plot_maze(ax5, np.load(os.path.join(load_path, 'q_explore_online.npy')), agent, colorbar=True, colormap='Purples', move=[14])
    ax5.set_title(r'Online discovery', fontsize=16)
    ax5.text(-0.1, 1.1, 'D', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax6 = fig.add_subplot(2,3,5)
    q_explore_replay_diff = np.load(os.path.join(load_path, 'q_explore_online_replay_diff.npy'))
    q_explore_replay_diff[q_explore_replay_diff == 0.] = np.nan
    plot_maze(ax6, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax6.set_title(r'Replay', fontsize=16)
    ax6.text(-0.1, 1.1, 'E', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax7 = fig.add_subplot(2,3,6)
    plot_maze(ax7, np.load(os.path.join(load_path, 'q_explore_online_replay.npy')), agent, colorbar=True, colormap='Purples', move=[14])
    ax7.set_title(r'Updated policy', fontsize=16)
    ax7.text(-0.1, 1.1, 'F', transform=ax7.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(load_path, 'fig_5.png'))
    plt.savefig(os.path.join(load_path, 'fig_5.svg'), transparent=True)

    return None

if __name__ == '__main__':
    main()