import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from utils import plot_maze

load_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/supp/supp3/data/1/0'))

def main(save_folder):

    with open(os.path.join(save_folder, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(14, 4), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(131)
    plot_maze(ax1, np.load(os.path.join(save_folder, 'q_mb.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)
    # ax1.set_ylabel(r'Initial $Q^{MF}$', fontsize=14)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(132)
    q_explore_replay_diff = np.load(os.path.join(save_folder, 'q_explore_replay_diff.npy'))
    q_explore_replay_diff[q_explore_replay_diff == 0.] = np.nan
    plot_maze(ax2, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax2.set_title(r'Exploratory replay', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(133)
    plot_maze(ax3, np.load(os.path.join(save_folder, 'q_explore_replay.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax3.set_title(r'Updated exploratory policy', fontsize=16)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(save_folder, 'supp_3.png'))
    plt.savefig(os.path.join(save_folder, 'supp_3.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main(load_path)