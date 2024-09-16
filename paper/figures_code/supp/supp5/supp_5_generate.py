import numpy as np
import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../../code/maze')))
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 'tolman123_nocheat'
env_file_path  = os.path.abspath(os.path.join(sys.path[0], '../../../code/mazes/' + env + '.txt'))
env_config     = load_env(env_file_path)

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 2,           
    'gamma'          : 0.9,
}

ag_config = {
    'alpha_r'        : 1,         # offline learning rate
    'horizon'        : 13,        # planning horizon (minus 1)
    'xi'             : 0.000001,  # EVB replay threshold
    'num_sims'       : 2000,      # number of MC simulations for need
    'sequences'      : False,      
    'env_name'       : env,       # gridworld name
}

save_path = os.path.abspath(os.path.join(sys.path[0], '../../../figures/supp/supp5'))

def main(save_folder):

    env_config['barriers'] = [1, 1, 0]

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    Q_MB  = agent._solve_mb(1e-5)

    np.save(os.path.join(save_folder, 'q_mb.npy'), Q_MB)
    np.savetxt(os.path.join(save_folder, 'q_mb.csv'), Q_MB, delimiter=',')

    agent.Q           = Q_MB.copy()
    agent.M           = np.array([[7, 2], [7, 2], [1, 0]])
    agent.belief_tree = agent._build_belief_tree()
    agent.pneed_tree  = agent._build_pneed_tree()

    with open(os.path.join(save_folder, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main(save_path)