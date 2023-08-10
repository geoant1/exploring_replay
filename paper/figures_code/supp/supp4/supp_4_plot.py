import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../../code/maze')))
from utils import plot_maze

load_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp3/data/'))
save_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp4'))

def main():

    with open(os.path.join(load_path, '0', '0', 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    priors = [[2, 2], [6, 2], [10, 2], [14, 2], [18, 2], [22, 2]]
    betas  = [1, 2, 4, 'greedy']
    sas    = [[14, 0], [20, 0], [19, 3], [18, 3], [24, 0], [30, 0], [31, 2], [32, 2]]
    probas = np.ones((len(priors), len(betas)))

    for idxb, beta in enumerate(betas):
        agent.beta = beta
        for idxp, prior in enumerate(priors):

            this_path = os.path.join(load_path, str(idxb), str(idxp))

            Q = np.load(os.path.join(this_path, 'q_explore_replay.npy'))

            for sa in sas:

                s, a = sa[0], sa[1]

                probas[idxp, idxb] *= agent._policy(Q[s, :])[a]

    fig = plt.figure(figsize=(4, 3), constrained_layout=True, dpi=100)

    colours = ['blue', 'orange', 'green', 'purple', 'red']

    for idxb, beta in enumerate(betas):
        plt.plot(range(len(priors)), probas[:, idxb], c=colours[idxb], label=r'$\beta=$%s'%beta)

    plt.legend(prop={'size':8})
    plt.ylabel('Exploration probability', fontsize=14)
    # plt.xticks(range(len(priors)), ['Beta(' + str(i).strip('[').strip(']') + ')' for i in priors], rotation=45)
    plt.xticks(range(len(priors)), [np.round(i[0]/(i[0]+i[1]), 2) for i in priors], rotation=45)
    plt.xlabel(r'$\mathbb{E}_b[p(open)]$', fontsize=14)

    plt.savefig(os.path.join(save_path, 'supp_4.png'))
    plt.savefig(os.path.join(save_path, 'supp_4.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main()