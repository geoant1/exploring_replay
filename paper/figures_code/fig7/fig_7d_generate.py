import numpy as np
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 't'
env_file_path  = os.path.abspath(os.path.join(sys.path[0], '../../code/mazes/' + env + '.txt'))
env_config     = load_env(env_file_path)

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 2,          
    'gamma'          : 0.9,
    'policy_type'    : 'softmax'
}

ag_config = {
    'alpha_r'        : 1,        # offline learning rate
    'horizon'        : 4,        # planning horizon (minus 1)
    'xi'             : 0.2,      # EVB replay threshold
    'num_sims'       : 2000,     # number of MC simulations for need
    'sequences'      : True,
    'max_seq_len'    : 2,
    'env_name'       : env,      # gridworld name
}

save_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig7/t_maze'))

def main(save_folder):

    env_config['barriers']  = [1]
    env_config['rew_value'] = [0, 0]

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    Q_MB  = agent.Q.copy()

    np.save(os.path.join(save_folder, 'q_init.npy'), Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_init.csv'), Q_MB, delimiter=',')

    a, b        = 7, 2
    agent.state = 7 # start state
    agent.M     = np.array([[a, b]])

    Q_history, _, _ = agent._replay()

    np.save(os.path.join(save_folder, 'q_explore_replay_before.npy'), agent.Q)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay_before.csv'), agent.Q, delimiter=',')

    np.save(os.path.join(save_folder, 'q_explore_replay_diff_before.npy'), agent.Q-Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay_diff_before.csv'), agent.Q-Q_MB, delimiter=',')

    agent.rew_value      = [0, 1]
    agent._init_reward()

    Q_history, _, _ = agent._replay()

    states = [1, 2, 3]
    Q = agent.Q_nans.copy()
    Q_tree = Q_history[-1]
    for hi in reversed(range(agent.horizon)):
        for k, val in Q_tree[hi].items():
            state  = val[0][1]
            if state in states:
                Q_vals = val[1]
                Q[state, :] = Q_vals[state, :]

    np.save(os.path.join(save_folder, 'q_explore_replay_after.npy'), Q)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay_after.csv'), Q, delimiter=',')

    np.save(os.path.join(save_folder, 'q_explore_replay_diff_after.npy'), Q-Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay_diff_after.csv'), Q-Q_MB, delimiter=',')

    with open(os.path.join(save_folder, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main(save_path)