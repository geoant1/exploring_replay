import numpy as np
import sys, os, shutil, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../../code/bandit')))
from belief_tree import Tree

# --- Specify parameters ---

# prior belief at the root
alpha_0, beta_0 = 13, 12
alpha_1, beta_1 = 2, 2

M = np.array([
    [alpha_0, beta_0],
    [alpha_1, beta_1]
])

# other parameters
p = {
    'arms':           ['unknown', 'unknown'],
    'root_belief':    M,
    'rand_init':      True,
    'gamma':          0.9,
    'xi':             0.0001,
    'beta':           4,
    'policy_type':    'softmax',
    'sequences':      True,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        5
}

# save path
save_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp4/data'))

# --- Main function for replay ---
def main(save_folder):
    
    num_trees = 10000

    seqs      = [True, False]

    np.random.seed(0)

    for seq in seqs:

        for tidx in range(num_trees):

            this_save_path = os.path.join(save_folder, str(tidx), str(seq), 'replay_data')

            if os.path.isdir(this_save_path):
                shutil.rmtree(this_save_path)
            os.makedirs(this_save_path)

            p['sequences'] = seq
            # initialise the agent
            tree   = Tree(**p)
        
            # do replay
            q_history, n_history, _, replays = tree.replay_updates()

            np.save(os.path.join(this_save_path, 'need_history.npy'), n_history)
            np.save(os.path.join(this_save_path, 'qval_history.npy'), q_history)
            np.save(os.path.join(this_save_path, 'replay_history.npy'), np.asarray(replays, dtype='object'))

            print('Done with tree %u/%u'%(tidx+1, num_trees))

    return None

if __name__ == '__main__':
    main(save_path)
