import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from utils import plot_maze

save_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig7/t_maze'))

def main(save_folder):

    with open(os.path.join(save_folder, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig  = plt.figure(figsize=(10, 4), constrained_layout=True, dpi=100)

    ax1  = fig.add_axes([0.02, 0.55, 0.45, 0.35])
    q_mb = np.load(os.path.join(save_folder, 'q_init.npy'))
    plot_maze(ax1, q_mb, agent, colorbar=True, colormap='Purples', move=[7])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)

    q_before = np.load(os.path.join(save_folder, 'q_explore_replay_diff_before.npy'))

    left  = [[1, 2], [2, 2]]
    right = [[2, 3], [3, 3]]

    lb = 0
    rb = 0

    for l in left:
        if q_before[l[0], l[1]] > 0:
            lb += 1
    for r in right:
        if q_before[r[0], r[1]] > 0:
            rb += 1

    ax2 = fig.add_axes([0.54, 0.6, 0.15, 0.3])
    ax2.bar([1, 2], [lb, rb])
    ax2.set_xticks([1, 2], ['uncued', 'cued'])
    ax2.set_ylim(0, 2.5)
    ax2.set_ylabel('Number of replays')
    ax2.set_title('Simulated data', fontsize=16)

    data = [0.0431, 0.0474]
    ax3  = fig.add_axes([0.8, 0.6, 0.15, 0.3])
    ax3.bar([1, 2], data)
    ax3.hlines(0.0380, xmin=1-0.4, xmax=1+0.4, color='r', linestyle='--')
    ax3.hlines(0.0456, xmin=2-0.4, xmax=2+0.4, color='r', linestyle='--')
    ax3.set_xticks([1, 2], ['uncued', 'cued'])
    ax3.set_ylim(0, 0.15)
    ax3.set_ylabel('Proportion of preplays')
    ax3.set_title('Real data', fontsize=16)

    np.savetxt(os.path.join(save_folder, 'cued_uncued_rest1.csv'), np.array([[lb, rb], data]))

    q_before = np.load(os.path.join(save_folder, 'q_explore_replay_diff_after.npy'))

    left  = [[1, 2], [2, 2]]
    right = [[2, 3], [3, 3]]

    lb = 0
    rb = 0

    for l in left:
        if q_before[l[0], l[1]] > 0:
            lb += 1
    for r in right:
        if q_before[r[0], r[1]] > 0:
            rb += 1

    ax3 = fig.add_axes([0.02, 0.1, 0.45, 0.35])
    plot_maze(ax3, q_before, agent, colorbar=True, colormap='Purples', move=[7])
    ax3.set_title(r'New exploratory policy', fontsize=16)

    ax4 = fig.add_axes([0.54, 0.1, 0.15, 0.3])
    ax4.bar([1, 2], [lb, rb])
    ax4.set_xticks([1, 2], ['uncued', 'cued'])
    ax4.set_ylim(0, 2.5)
    ax4.set_ylabel('Number of replays')

    data = [0.0441, 0.0737]
    ax3  = fig.add_axes([0.8, 0.1, 0.15, 0.3])
    ax3.bar([1, 2], data)
    ax3.hlines(0.0422, xmin=1-0.4, xmax=1+0.4, color='r', linestyle='--')
    ax3.hlines(0.0486, xmin=2-0.4, xmax=2+0.4, color='r', linestyle='--')
    ax3.set_xticks([1, 2], ['uncued', 'cued'])
    ax3.set_ylim(0, 0.15)
    ax3.set_ylabel('Proportion of preplays')

    np.savetxt(os.path.join(save_folder, 'cued_uncued_rest2.csv'), np.array([[lb, rb], data]))

    plt.savefig(os.path.join(save_folder, 'fig_7_tmaze.png'))
    plt.savefig(os.path.join(save_folder, 'fig_7_tmaze.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main(save_path)