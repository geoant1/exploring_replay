import os, pickle, glob, shutil
import matplotlib.pyplot as plt
from tex_tree import generate_value_diff_tree
import numpy as np
from copy import deepcopy

def plot_root_values(data_folder):

    with open(os.path.join(data_folder, 'tree.pkl'), 'rb') as f:
        tree = pickle.load(f)

    qval_history = np.load(os.path.join(data_folder, 'replay_data', 'qval_history.npy'), allow_pickle=True)

    root_values   = []
    policy_values = []

    for i in qval_history:
        qval_tree      = i
        policy_values += [tree.evaluate_policy(i)]

        qvals          = qval_tree[0][0]
        root_values   += [tree._value(qvals)]

    v_full = np.max(tree.full_updates()[0][0])

    plt.figure(figsize=(6, 5), dpi=100, constrained_layout=True)
    
    for i in [1, 2]:
        plt.subplot(2, 1, i)

        if i == 1:
            plt.plot(root_values)
            plt.ylabel(r'$V(b_{\rho})$', fontsize=12)
        else:
            plt.plot(policy_values)
            plt.ylabel(r'$V^{\pi}$', fontsize=12)

        plt.axhline(v_full, linestyle='--', color='k', alpha=0.7, label='Optimal value')
        plt.tick_params(axis='y', labelsize=10)

        if i == 1:
            plt.xticks([])
        else:
            plt.xlabel('Number of updates', fontsize=12)
            plt.xticks(range(len(root_values)), range(len(root_values)), fontsize=10)
        
        plt.xlim(0, len(qval_history)-1)
        plt.ylim(0, v_full+2)
        plt.legend(prop={'size':9})

    plt.savefig(os.path.join(data_folder, 'root_values.png'))
    plt.savefig(os.path.join(data_folder, 'root_values.svg'), transparent=True)
    plt.close()

def plot_multiple(data_folder, M, P, R, nreps, R_true, horizons, xis, betas):

    fig, axes = plt.subplots(6, 2, figsize=(9, 18), dpi=100, constrained_layout=True, gridspec_kw={'height_ratios':[2, 2, 1, 2, 2, 1]})
    # plt.suptitle('alpha0 = %u, beta0 = %u, alpha1 = %u, beta1 = %u'%(alpha_0, beta_0, alpha_1, beta_1), fontsize=14)

    for hidx, h in enumerate(horizons):

        if (hidx == 0) or (hidx == 1): 
            axv = axes[0, hidx%2]
            axp = axes[1, hidx%2]
            axr = axes[2, hidx%2]
        else:
            axv = axes[3, hidx%2]
            axp = axes[4, hidx%2]
            axr = axes[5, hidx%2]

        for bidx, beta in enumerate(betas): 
            
            axv.plot(R[hidx, bidx, ::-1], label='Beta %.1f'%beta)
            axv.scatter(range(len(xis)), R[hidx, bidx, ::-1])

            axp.plot(P[hidx, bidx, ::-1], label='Beta %.1f'%beta)
            axp.scatter(range(len(xis)), P[hidx, bidx, ::-1])
            
            axr.plot(nreps[hidx, bidx, ::-1], label='Beta %.1f'%beta)
            axr.scatter(range(len(xis)), nreps[hidx, bidx, ::-1])

            if bidx == (len(betas) - 1):

                print(hidx, R_true)
                axv.axhline(R_true[hidx], linestyle='--', color='k', alpha=0.7, label='Optimal value')
                axp.axhline(R_true[hidx], linestyle='--', color='k', alpha=0.7, label='Optimal value')
            
                axv.legend(prop={'size': 13})
                # axp.legend(prop={'size': 13})
                # axr.legend(prop={'size': 13})

                axv.set_ylabel('Root value', fontsize=17)
                axv.set_ylim(0, np.max(R_true)+0.1)
                axv.set_title('Horizon %u'%(h-1), fontsize=18)
                axv.tick_params(axis='y', labelsize=13)

                axp.set_ylabel('Policy value', fontsize=17)
                axp.set_ylim(0, np.max(R_true)+0.1)
                axp.tick_params(axis='y', labelsize=13)

                axr.set_ylabel('Number of updates', fontsize=17)
                axr.tick_params(axis='y', labelsize=13)
                axr.set_ylim(0, np.nanmax(nreps)+6)

                axr.set_xlabel(r'$\xi$', fontsize=17)
                axr.set_xticks(range(R.shape[2]), ['%.4f'%i for i in xis[::-1]], rotation=60, fontsize=13)

                axv.set_xticks([])
                axp.set_xticks([])

    file_name = 'alpha0%u_beta0%u_alpha1%u_beta1%u_complete'%(M[0, 0], M[0, 1], M[1, 0], M[1, 1])
    np.save(os.path.join(data_folder, file_name + '.npy'), R)
    plt.savefig(os.path.join(data_folder, file_name + '.svg'), transparent=True)
    plt.savefig(os.path.join(data_folder, file_name + '.png'))
    plt.close()

def analyse(data_path):

    save_path = os.path.join(data_path, 'analysis')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    else: pass

    num_trees = max([int(i.split('/')[-1]) for i in glob.glob(os.path.join(data_path, '*'))])

    P     = np.zeros(num_trees)
    R     = np.zeros(num_trees)
    S     = np.zeros(num_trees)
    N     = np.zeros((num_trees, 2))

    for t in range(num_trees+1):
        
        this_folder_seqs   = os.path.join(data_path, str(t), 'seqs')
        this_folder_noseqs = os.path.join(data_path, str(t), 'noseqs')
        
        with open(os.path.join(this_folder_seqs, 'tree.pkl'), 'rb') as f:
            tree = pickle.load(f)

        qvals         = tree.qval_tree[0][0]
        v_replay      = tree._value(qvals)
        eval_pol      = tree.evaluate_policy(tree.qval_tree)

        P[t]  = eval_pol
        R[t]  = v_replay

        replays  = np.load(os.path.join(this_folder_seqs, 'replay_data', 'replay_history.npy'), allow_pickle=True)
        num_seqs = 0
        rev      = 0
        fow      = 0
        for replay in replays[1:]:
            num_seqs += len(replay[0])
            if len(replay[0]) > 1:
                if replay[0][0] > replay[0][1]:
                    rev += 1
                else:
                    fow += 1
        S[t]    = fow/(fow + rev)
        N[t, 0] = num_seqs

        replays    = np.load(os.path.join(this_folder_noseqs, 'replay_data', 'replay_history.npy'), allow_pickle=True)
        num_noseqs = len(replays[1:])
        N[t, 1]    = num_noseqs


    v_full    = tree.full_updates()
    R_true    = np.max(v_full[0][0])

    os.makedirs(save_path)

    np.save(os.path.join(save_path, 'eval_pol.npy'), P)
    np.save(os.path.join(save_path, 'root_pol.npy'), R)
    np.save(os.path.join(save_path, 'full_upd.npy'), R_true)

    fig = plt.figure(figsize=(6, 6))
    plt.bar([0.5], np.mean(P), width=0.3)
    plt.scatter([0.5]*len(P), P)
    plt.axhline(R_true, color='k', label='Optimal')
    plt.xlim(0, 1)
    plt.ylabel('policy value')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'policy_value.png'))

    fig = plt.figure(figsize=(6, 6))
    plt.bar([0.5], np.mean(S), width=0.3)
    plt.scatter([0.5]*len(S), S)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('prop of fwd sequences')
    plt.savefig(os.path.join(save_path, 'fwd_rev_prop.png'))

    fig = plt.figure(figsize=(6, 6))
    plt.bar([0.35], np.mean(N[0, :]), width=0.25)
    plt.bar([0.65], np.mean(N[1, :]), width=0.25)
    for i in range(N.shape[0]):
        plt.plot([0.35, 0.65], N[i, :], c='k', alpha=0.7)
    plt.xlim(0, 1)
    plt.xticks([0.35, 0.65], ['Sequences', 'Single'])
    plt.ylabel('Total number of updates')
    plt.savefig(os.path.join(save_path, 'num_updates.png'))

    idcs_greater = np.argwhere(S >= S.mean()).flatten()
    idcs_smaller = np.argwhere(S < S.mean()).flatten()

    for idx, t in enumerate(idcs_greater):
        this_folder_noseqs   = os.path.join(data_path, str(t), 'noseqs')
        q_init = np.load(os.path.join(this_folder_noseqs, 'replay_data', 'qval_history.npy'), allow_pickle=True)[0]
        if idx == 0:
            q_greater = deepcopy(q_init)
            for hi in range(len(q_greater.keys())):
                for idx, vals in q_greater[hi].items():
                    q_greater[hi][idx] = vals/len(idcs_greater)
        else:
            for hi in range(len(q_greater.keys())):
                for idx, vals in q_greater[hi].items():
                    q_greater[hi][idx] += q_init[hi][idx]/len(idcs_greater)

    for idx, t in enumerate(idcs_smaller):
        this_folder_noseqs   = os.path.join(data_path, str(t), 'noseqs')
        q_init = np.load(os.path.join(this_folder_noseqs, 'replay_data', 'qval_history.npy'), allow_pickle=True)[0]
        if idx == 0:
            q_smaller = deepcopy(q_init)
            for hi in range(len(q_smaller.keys())):
                for idx, vals in q_smaller[hi].items():
                    q_smaller[hi][idx] = vals/len(idcs_smaller)
        else:
            for hi in range(len(q_smaller.keys())):
                for idx, vals in q_smaller[hi].items():
                    q_smaller[hi][idx] += q_init[hi][idx]/len(idcs_smaller)

    file_name = os.path.join(save_path, 'init_val_diff.tex')
    generate_value_diff_tree(q_greater, q_smaller, file_name)

    return None