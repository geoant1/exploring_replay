import numpy as np
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))

data_path   = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig3/data/'))
save_path   = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig3'))

num_moves   = 2000

def main(data_folder, save_folder):
        
    with open(os.path.join(data_folder, 'yes_forgetting', '0', 'ag.pkl'), "rb") as f:
        agent = pickle.load(f)

    num_seeds = len([i for i in os.listdir(os.path.join(data_folder, 'yes_forgetting')) if os.path.isdir(os.path.join(data_folder, 'yes_forgetting', i))])

    S  = [np.zeros((num_seeds, agent.num_states)), np.zeros((num_seeds, agent.num_states))]
    G  = [np.full((num_seeds, num_moves, agent.num_states, agent.num_actions), np.nan), np.full((num_seeds, num_moves, agent.num_states, agent.num_actions), np.nan)]
    N  = [np.full((num_seeds, num_moves, agent.num_states), np.nan), np.full((num_seeds, num_moves, agent.num_states), np.nan)]

    for idx, bounds in enumerate([[0, num_moves], [num_moves, num_moves*2]]):

        for seed in range(num_seeds):
            for file in range(bounds[0], bounds[1]):
                    
                data         = np.load(os.path.join(data_folder, 'yes_forgetting', str(seed), 'Q_%u.npz'%file), allow_pickle=True)
                move         = data['move']
                s            = int(move[0])
                S[idx][seed, s] += 1

                if 'gain_history' in data.files:
                    gain_history = data['gain_history']
                    need_history = data['need_history']
                    for gidx in range(len(gain_history)):
                        for st in np.delete(range(agent.num_states), agent.nan_states):
                            need_value     = need_history[gidx][st]
                            if need_value == np.nanmax([N[idx][seed, file%num_moves, st], need_value]):
                                N[idx][seed, file%num_moves, st] = need_value
                            for at in range(agent.num_actions):
                                gain_value = gain_history[gidx][st, at]
                                if ~np.isnan(gain_value):
                                    if gain_value == np.nanmax([G[idx][seed, file%num_moves, st, at], gain_value]):
                                        G[idx][seed, file%num_moves, st, at] = gain_value

        G[idx]  = np.nanmean(G[idx], axis=(0, 1))
        N[idx]  = np.nanmean(N[idx], axis=(0, 1))
        S[idx]  = np.mean(S[idx],    axis=0)

    np.save(os.path.join(save_folder, 'gain.npy'), G)
    np.save(os.path.join(save_folder, 'need.npy'), N)
    np.save(os.path.join(save_folder, 'states.npy'), S)

    return None

if __name__ == '__main__':
    main(data_path, save_path)