import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(sys.path[0], "../../../code/bandit")))
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp

# --- Specify parameters ---

# save path
save_path = os.path.abspath(os.path.join(sys.path[0], "../../../figures/supp/supp4/"))

num_trees = 10000
seqs = [True, False]


# --- Main function for replay ---
def main(save_folder):

    prop_matrix = np.zeros((num_trees, 2))
    num_matrix = np.zeros((num_trees, 2))

    for tidx in range(num_trees):

        for sidx, seq in enumerate(seqs):

            data = np.load(
                os.path.join(
                    save_folder,
                    "data",
                    str(tidx),
                    str(seq),
                    "replay_data",
                    "replay_history.npy",
                ),
                allow_pickle=True,
            )
            # proportion of single-step and sequence replays

            num_fwd_prop = 0
            num_rev_prop = 0

            num_evnts = 0

            num_replays = len(data) - 1
            num_seqs = 0

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
                prop_matrix[tidx, sidx] = num_seqs / num_replays
                if num_rev_prop == 0:
                    prop_matrix[tidx, sidx] = 1
                else:
                    prop_matrix[tidx, sidx] = num_fwd_prop / (
                        num_rev_prop + num_fwd_prop
                    )

                num_matrix[tidx, sidx] = num_evnts
            else:
                prop_matrix[tidx, sidx] = np.nan

    np.save(os.path.join(save_folder, "num_matrix.npy"), num_matrix)
    np.savetxt(os.path.join(save_folder, "num_matrix.csv"), num_matrix, delimiter=",")

    np.save(os.path.join(save_folder, "prop_matrix.npy"), prop_matrix)
    np.savetxt(os.path.join(save_folder, "prop_matrix.csv"), prop_matrix, delimiter=",")

    tp, pp = ttest_1samp(prop_matrix[:, 0], 0)
    plt.title("t = %.2f, p=%.2e" % (tp, pp))

    tn, pn = ttest_ind(num_matrix[:, 0], num_matrix[:, 1])
    plt.title("t = %.2f, p=%.2e" % (tn, pn))

    with open(os.path.join(save_folder, "stats.txt"), "w") as f:
        f.write("Proportion: t=%.2f, p=%.2e\n" % (tp, pp))
        f.write("Number:     t=%.2f, p=%.2e\n" % (tn, pn))

    return None


if __name__ == "__main__":
    main(save_path)
