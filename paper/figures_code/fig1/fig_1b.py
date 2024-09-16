import numpy as np
import os, shutil, pickle, sys
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/bandit')))
from belief_tree import Tree
from tex_tree import generate_big_tex_tree

# --- Specify parameters ---

# prior belief at the root

mu_0 = 0.50
mu_1 = 0.51

M = {
    0: mu_0, 
    1: mu_1
    }

# other parameters
p = {
    'arms':           ['known', 'known'],
    'root_belief':    M,
    'rand_init':      False,
    'gamma':          0.9,
    'xi':             0.0001,
    'beta':           2,
    'policy_type':    'softmax',
    'sequences':      False,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        2
}

# save path
save_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig1/data/b'))

# --- Main function for replay ---
def main(save_folder, params, plot_tree=False):
    
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    else: pass
    os.makedirs(save_folder)
    
    tree   = Tree(**params)
    qval_history, need_history, gain_history, replay_history = tree.replay_updates()

    print('Number of replays: %u'%(len(replay_history)-1))
    print('Policy value: %.2f'%tree.evaluate_policy(tree.qval_tree))

    os.mkdir(os.path.join(save_folder, 'replay_data'))
    np.save(os.path.join(save_folder, 'replay_data', 'qval_history.npy'), qval_history)
    np.save(os.path.join(save_folder, 'replay_data', 'need_history.npy'), need_history)
    np.save(os.path.join(save_folder, 'replay_data', 'gain_history.npy'), gain_history)
    np.save(os.path.join(save_folder, 'replay_data', 'replay_history.npy'), replay_history)

    if plot_tree:
        os.mkdir(os.path.join(save_folder, 'tree'))
        for idx in range(len(replay_history)):
            these_replays  = replay_history[:idx+1]
            this_save_path = os.path.join(save_folder, 'tree', 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(tree, these_replays, qval_history[idx], need_history[idx], this_save_path, tree_height=3)

    with open(os.path.join(save_folder, 'tree.pkl'), 'wb') as f:
        pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)
    
    # save params
    with open(os.path.join(save_folder, 'params.txt'), 'w') as f:
        for k, v in p.items():
            f.write(k)
            f.write(':  ')
            f.write(str(v))
            f.write('\n')

    return None

if __name__ == '__main__':
    main(save_path, p, plot_tree=True)