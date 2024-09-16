import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from utils import plot_maze

save_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig7'))

def main(save_folder):

    with open(os.path.join(save_folder, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(231)
    q_mb = np.load(os.path.join(save_folder, 'q_mb.npy'))
    plot_maze(ax1, q_mb, agent, colorbar=True, colormap='Purples', move=[38])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(232)
    q_explore_replay2_diff = np.load(os.path.join(save_folder, 'q_explore_replay2_diff.npy'))
    plot_maze(ax2, q_explore_replay2_diff, agent, colorbar=True, colormap='Purples', move=[38])
    ax2.set_title(r'Exploratory replay', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(235)
    q_explore_replay1_diff = np.load(os.path.join(save_folder, 'q_explore_replay1_diff.npy'))
    plot_maze(ax3, q_explore_replay1_diff, agent, colorbar=True, colormap='Purples', move=[38])
    ax3.set_title(r'Exploratory replay', fontsize=16)
    # ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax4 = fig.add_subplot(233)
    q_explore_replay2 = np.load(os.path.join(save_folder, 'q_explore_replay2.npy'))
    plot_maze(ax4, q_explore_replay2, agent, colorbar=True, colormap='Purples', move=[38])
    ax4.set_title(r'New exploratory policy', fontsize=16)
    ax4.text(-0.1, 1.1, 'C', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(save_folder, 'fig_7.png'))
    plt.savefig(os.path.join(save_folder, 'fig_7.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main(save_path)