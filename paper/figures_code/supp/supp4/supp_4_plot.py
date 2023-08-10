import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../../code/bandit')))
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp

# --- Specify parameters ---

# save path
save_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp4/'))

num_trees = 100
seqs      = [True, False]

# --- Main function for replay ---
def main(save_folder):

    prop_matrix = np.zeros((num_trees, 2))
    num_matrix  = np.zeros((num_trees, 2))

    for tidx in range(num_trees):
        
        for sidx, seq in enumerate(seqs):

            data = np.load(os.path.join(save_folder, 'data', str(tidx), str(seq), 'replay_data', 'replay_history.npy'), allow_pickle=True)
                # proportion of single-step and sequence replays

            num_fwd_prop = 0
            num_rev_prop = 0

            num_evnts    = 0

            num_replays = len(data) - 1
            num_seqs    = 0

            if seq == True:
                for replay in data[1:]:
                    if len(replay[0]) > 1:
                        num_seqs += 1
                        # forward or reverse
                        if replay[0][0] > replay[0][1]:
                            num_rev_prop += 1
                        else:
                            num_fwd_prop += 1

                    num_evnts += len(replay[0])
            else:
                num_evnts = num_replays

            if num_replays > 0:
                prop_matrix[tidx, sidx] = num_seqs/num_replays
                if num_rev_prop == 0:
                    prop_matrix[tidx, sidx] = 1
                else:
                    prop_matrix[tidx, sidx]  = num_fwd_prop/(num_rev_prop + num_fwd_prop)

                num_matrix[tidx, sidx] = num_evnts
            else:
                prop_matrix[tidx, sidx] = np.nan

    plt.figure(figsize=(8, 4), dpi=100, constrained_layout=True)
    plt.subplot(121)
    plt.bar([1], np.nanmean(prop_matrix[:, 0]), width=0.3, facecolor='k', alpha=0.6)
    plt.scatter([1]*prop_matrix.shape[0], prop_matrix[:, 0], c='k')
    plt.axhline(0, c='k')
    plt.ylim(0.0, 1.05)
    plt.xlim(0.5, 1.5)
    plt.xticks([])
    plt.ylabel('Proportion of forward to reverse sequences', fontsize=12)
    tp, pp = ttest_1samp(prop_matrix[:, 0], 0)
    plt.title('t = %.2f, p=%.2e'%(tp, pp))

    plt.subplot(122)
    plt.bar([1], np.nanmean(num_matrix[:, 0]), width=0.7, facecolor='purple', alpha=0.6)
    plt.scatter([1]*num_trees, num_matrix[:, 0], c='purple')
    plt.bar([2], np.nanmean(num_matrix[:, 1]), width=0.7, facecolor='green', alpha=0.6)
    plt.scatter([2]*num_trees, num_matrix[:, 1], color='green')
    for i in range(num_trees):
        plt.plot([1, 2], [num_matrix[i, 0], num_matrix[i, 1]], c='k')

    tn, pn = ttest_ind( num_matrix[:, 0], num_matrix[:, 1])
    plt.title('t = %.2f, p=%.2e'%(tn, pn))

    plt.xticks([1, 2], ['seqs', 'noseqs'], fontsize=12)
    plt.xlim(0, 3)
    plt.ylim(0, np.max(num_matrix[:] + 8))
    plt.axhline(0, c='k')
    plt.ylabel('Number of replayed actions', fontsize=12)
    plt.savefig(os.path.join(save_folder, 'supp_4.png'))
    plt.savefig(os.path.join(save_folder, 'supp_4.svg'), transparent=True)
    plt.close()

    with open(os.path.join(save_folder, 'stats.txt'), 'w') as f:
        f.write('Proportion: t=%.2f, p=%.2e\n'%(tp, pp))
        f.write('Number:     t=%.2f, p=%.2e\n'%(tn, pn))

    return None

if __name__ == '__main__':
    main(save_path)
