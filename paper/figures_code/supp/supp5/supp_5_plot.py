import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../../code/maze')))
from utils import plot_maze, plot_need
import matplotlib.pyplot as plt

load_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp5'))

def main():

    with open(os.path.join(load_path, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    updates = agent._generate_single_updates()

    fig  = plt.figure(figsize=(18, 16), dpi=100)

    ax   = plt.subplot(3, 3, 4)
    plot_maze(ax, agent.Q, agent, colormap='Purples')
    idcs = [[4, 0], [8, 4], [11, 2]]

    for plt_idx, (hip, kp) in enumerate(idcs):

        b = agent.belief_tree[hip][kp][0][0]

        need = np.full(agent.num_states, np.nan)

        for idx, upd in enumerate(updates):
            hi, k = upd[0], upd[1]
            if hi == hip and k == kp:
                print(hi, k)
                sp, ap   = upd[2][0, :]
                gain_val = upd[-3][0][hi[0]][k[0]]
                print(gain_val)
                need[sp] = agent.pneed_tree[hi[0]][k[0]]
                print(need[sp])
                break

        ax   = plt.subplot(3, 3, (plt_idx*3)+2)
        plot_need(ax, need, agent, normalise=False, colormap='Oranges', vmin=0, vmax=0.12)

        Gain = agent.Q_nans.copy()
        Gain[sp, ap] = gain_val
        ax   = plt.subplot(3, 3, (plt_idx*3)+3)
        plot_maze(ax, Gain, agent)

    plt.savefig(os.path.join(load_path, 'supp_5.png'))
    plt.savefig(os.path.join(load_path, 'supp_5.svg'), transparent=True)


    return None

if __name__ == '__main__':
    main()