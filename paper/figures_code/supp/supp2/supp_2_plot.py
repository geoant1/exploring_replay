import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../../code/maze')))
from utils import plot_maze

load_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp2/data/1/0'))
save_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp2/'))

def main():

    with open(os.path.join(load_path, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(14, 4), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(131)
    q_mb = np.load(os.path.join(load_path, 'q_mb.npy'))
    np.savetxt(os.path.join(save_path, 'q_mb.csv'), q_mb, delimiter=',')
    plot_maze(ax1, q_mb, agent, colorbar=True, colormap='Purples', move=[38])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)
    # ax1.set_ylabel(r'Initial $Q^{MF}$', fontsize=14)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(132)
    q_explore_replay_diff = np.load(os.path.join(load_path, 'q_explore_replay_diff.npy'))
    np.savetxt(os.path.join(save_path, 'q_explore_replay_diff.csv'), q_explore_replay_diff, delimiter=',')
    q_explore_replay_diff[q_explore_replay_diff == 0.] = np.nan
    plot_maze(ax2, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax2.set_title(r'Exploratory replay', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(133)
    q_explore_replay = np.load(os.path.join(load_path, 'q_explore_replay.npy'))
    np.savetxt(os.path.join(save_path, 'q_explore_replay.csv'), q_explore_replay, delimiter=',')
    plot_maze(ax3, q_explore_replay, agent, colorbar=True, colormap='Purples', move=[38])
    ax3.set_title(r'Updated exploratory policy', fontsize=16)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(save_path, 'supp_2.png'))
    plt.savefig(os.path.join(save_path, 'supp_2.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main()