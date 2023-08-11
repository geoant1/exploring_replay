import numpy as np
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 'tolman1234'
env_file_path  = os.path.abspath(os.path.join(sys.path[0], '../../code/mazes/' + env + '.txt'))
env_config     = load_env(env_file_path)

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 2,      
    'gamma'          : 0.9,
}

ag_config = {
    'alpha_r'        : 1,        # offline learning rate
    'horizon'        : 6,       # planning horizon (minus 1)
    'xi'             : 0.000001,    # EVB replay threshold
    'num_sims'       : 2000,     # number of MC simulations for need
    'sequences'      : True,
    'max_seq_len'    : 4,
    'env_name'       : env,      # gridworld name
}

save_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig7'))

def main(save_folder):

    env_config['barriers'] = [1, 1, 1, 1]

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    Q_MB  = agent._solve_mb(1e-5)

    np.save(os.path.join(save_folder, 'q_mb.npy'), Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_mb.csv'), Q_MB, delimiter=',')

    a1, b1          = 7, 2
    a2, b2          = 7, 2
    agent.state     = 38          # start state
    agent.M         = np.array([[a1, b1], [a2, b2]])
    agent.Q         = Q_MB.copy() # set MF Q values
    Q_history, _, _ = agent._replay()

    belief_tree = Q_history[-1]

    Q1 = Q_MB.copy()
    for hi in range(agent.horizon):
        for k, v in belief_tree[hi].items():
            if np.array_equal(agent.M, v[0][0]):
                s = v[0][1]
                q = v[1]
                Q1[s, :] = q[s, :].copy()

    new_M = agent.M.copy()
    new_M[1, :] = [1, 0]
    Q2 = Q_MB.copy()
    for hi in range(agent.horizon):
        for k, v in belief_tree[hi].items():
            if np.array_equal(new_M, v[0][0]):
                s = v[0][1]
                q = v[1]
                Q2[s, :] = q[s, :].copy()

    np.save(os.path.join(save_folder, 'q_explore_replay1.npy'), Q1)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay1.csv'), Q1, delimiter=',')

    np.save(os.path.join(save_folder, 'q_explore_replay1_diff.npy'), Q1-Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay1_diff.csv'), Q1-Q_MB, delimiter=',')

    np.save(os.path.join(save_folder, 'q_explore_replay2.npy'), Q2)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay2.csv'), Q2, delimiter=',')

    np.save(os.path.join(save_folder, 'q_explore_replay2_diff.npy'), Q2-Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay2_diff.csv'), Q2-Q_MB, delimiter=',')

    np.save(os.path.join(save_folder, 'q_explore_replay_diff.npy'), agent.Q-Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_explore_replay_diff.csv'), agent.Q-Q_MB, delimiter=',')

    with open(os.path.join(save_folder, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main(save_path)