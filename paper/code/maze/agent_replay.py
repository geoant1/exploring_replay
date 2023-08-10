from environment import Environment
from panagent import PanAgent
import numpy as np
from copy import deepcopy, copy
import os, shutil, ast

class AgentPOMDP(PanAgent, Environment):

    def __init__(self, *configs):
        
        '''
        ----
        configs is a list containing 
            [0] panagent, [1] agent, [2] environment parameters

        panagent parameters:
            alpha          -- on-line MF learning rate
            beta           -- inverse temperature
            gamma          -- discount factor
            mf_forget      -- MF forgetting rate

        agent parameters:
            alpha_r        -- off-line replay learning rate
            horizon        -- planning horizon
            xi             -- EVB threshold
            num_sims       -- number of particles for need
            sequences      -- True/False
            max_seq_length -- maximal sequence length
            env_name       -- environment name
            barriers       -- which barriers are present/absent

        returns:
            None
        ----
        '''
        
        pag_config = configs[0]
        ag_config  = configs[1]
        env_config = configs[2]
        
        Environment.__init__(self, **env_config)
        PanAgent.__init__(self, **pag_config)
        self.__dict__.update(**ag_config)

        self.state = self.start_state

        # initialise MF Q values
        self._init_q_values()

        # initialise prior
        self.M = np.ones((len(self.barriers), 2))

        return None

    def _find_belief(self, z):

        '''
        ---
        determine whether belief state z exists in the planning tree

        parameters:
            z -- belief state

        returns:
            True/False, Q values
        ---
        '''

        b = z[0]
        s = z[1]

        for hi in range(self.horizon):
            for _, vals in self.belief_tree[hi].items():
                    
                if (s == vals[0][1]) and np.array_equal(b, vals[0][0]):
                    q = vals[1]

                    return True, q

        return False, None

    def _simulate_trajs(self):

        '''
        ---
        simulate forward histrories from the agent's current belief state
        
        parameters:
            None
        
        returns:
            his 
        ---
        '''

        his = {hi:{} for hi in range(self.horizon)}

        for sim in range(self.num_sims):
            s     = self.state # initial state
            b     = self.M     # current belief
            d     = 1          # counter

            # check if the current belief state is already in the belief tree
            hi, k, _ = self._check_belief_exists(self.belief_tree, [b, s])
            # if not then add it
            if k not in his[hi].keys():
                his[hi][k] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]
            
            his[hi][k][0][sim] = 1 # need
            his[hi][k][1][sim] = 0 # number of steps to reach this belief state

            # start particle simulations
            while ((self.gamma**d) > 1e-4):
                
                # check if the current belief state is within the subject's horizon reach
                check, q = self._find_belief([b, s])

                # terminate the particle if beyond
                if not check:
                    break
                
                # choose an action to perform at this belief state
                qvals = q[s, :]
                probs = self._policy(qvals)
                a     = np.random.choice(range(self.num_actions), p=probs)
                
                # check whether the subject is uncertain about this action outcome
                bidx = self._check_uncertain([s, a])
                # if yes
                if bidx is not None:

                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                    s1l, _ = self._get_new_state(s, a, unlocked=False)

                    # sample next state
                    bp = b[bidx, 0]/np.sum(b[bidx, :])
                    s1 = np.random.choice([s1u, s1l], p=[bp, 1-bp])

                    # update belief based on the transition
                    b = self._belief_plan_update(b, bidx, s, s1)
                # if not
                else:
                    s1, _  = self._get_new_state(s, a, unlocked=False)

                # check if this new belief state is within the subject's horizon reach
                hi, k, check = self._check_belief_exists(self.belief_tree, [b, s1])

                # terminate the particle if beyond
                if not check:
                    break
                
                # otherwise add 
                if k not in his[hi].keys():
                    his[hi][k] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]

                # update the associated values
                curr_val = his[hi][k][0][sim]
                if np.isnan(curr_val):
                    his[hi][k][0][sim] = self.gamma**d
                    his[hi][k][1][sim] = d
                
                s  = s1
                d += 1

        return his

    def _belief_plan_update(self, M, idx, s, s1):

        '''
        ---
        perform bayesian belief update

        parameters:
            M   -- current belief 
            idx -- belief about which barrier to update
            s   -- initial physical state
            s1  -- final physical state

        returns
            M_out
        ---
        '''

        M_out = M.copy()

        # unsuccessful 
        if s == s1:
            M_out[idx, 1] = 1
            M_out[idx, 0] = 0
        # successful
        else:
            M_out[idx, 0] = 1
            M_out[idx, 1] = 0

        return M_out

    def _check_belief_exists(self, btree, z):

        '''
        ---
        check if the belief state z already exists in the tree

        parameters:
            btree -- current belief tree
            z     -- belief state

        returns:
            horizon, idx, True/False
        ---
        '''

        b = z[0]
        s = z[1]

        for hi in range(self.horizon):
            this_tree = btree[hi]
            for k, vals in this_tree.items():
                if np.array_equal(vals[0][0], b) and (vals[0][1] == s):
                    return hi, k, True

        return None, None, False

    def _build_belief_tree(self):
        
        '''
        ---
        build a tree with future belief states up to horizon self.horizon

        parameters:
            None

        returns:
            btree -- {hi:{k: vals}}, hi: horizon, k: unique id, vals: [[b, s], Q, [a, hi, idx]]
        ---
        '''

        # each horizon hosts a number of belief states
        btree = {hi:{} for hi in range(self.horizon)}

        # create a merged tree -- one single tree for all information states
        idx = 0
        btree[0][idx] = [[self.M.copy(), self.state], self.Q.copy(), []]

        for hi in range(1, self.horizon):
            
            # unique index for each belief
            idx = 0

            if len(btree[hi-1]) == 0:
                break

            for prev_idx, vals in btree[hi-1].items():
                
                # retrieve previous belief information
                b = vals[0][0].copy()
                s = vals[0][1]
                q = vals[1].copy()

                # terminate at the goal state
                if s in self.goal_states:
                    continue
                
                for a in range(self.num_actions):
                    if ~np.isnan(self.Q[s, a]):
                        
                        bidx = self._check_uncertain([s, a])
                        if bidx is not None:

                            # if it's the uncertain state+action then this generates 
                            # two distinct beliefs
                            # first when the agent transitions through
                            s1u, _ = self._get_new_state(s, a, unlocked=True)
                            b1u    = self._belief_plan_update(b, bidx, s, s1u)

                            # second when it doesn't
                            s1l    = s
                            b1l    = self._belief_plan_update(b, bidx, s, s)

                            # check if these new beliefs already exist
                            hiu, idxu, checku = self._check_belief_exists(btree, [b1u, s1u])
                            hil, idxl, checkl = self._check_belief_exists(btree, [b1l, s1l])

                            # if both don't then add them to the belief tree
                            # and the new keys to the previous belief
                            to_add = [] 
                            if not checku and not checkl:
                                if b[bidx, 0] != 0:
                                    btree[hi][idx]        = [[b1u.copy(), s1u], q.copy(), []]
                                    to_add               += [[a, hi, idx]]
                                if b[bidx, 1] != 0:
                                    btree[hi][idx+1]      = [[b1l.copy(), s1l], q.copy(), []]
                                    to_add               += [[a, hi, idx+1]]
                                
                                btree[hi-1][prev_idx][2] += [to_add]
                                idx                      += len(to_add)

                            elif not checku and checkl:
                                if b[bidx, 0] != 0:
                                    btree[hi][idx]        = [[b1u.copy(), s1u], q.copy(), []]
                                    to_add               += [[a, hi, idx]]
                                    idx                  += 1
                                to_add                   += [[a, hil, idxl]]
                                btree[hi-1][prev_idx][2] += [to_add]

                            elif checku and not checkl:
                                to_add += [[a, hiu, idxu]]
                                if b[bidx, 1] != 0:
                                    btree[hi][idx]        = [[b1l.copy(), s1l], q.copy(), []]
                                    to_add               += [[a, hi, idx]]
                                    idx                  += 1
                                btree[hi-1][prev_idx][2] += [to_add]
                            else:
                                # if both exist then add their existing keys 
                                to_add = [[a, hiu, idxu], [a, hil, idxl]]
                                if (to_add not in btree[hi-1][prev_idx][2]):
                                    btree[hi-1][prev_idx][2] += [to_add]
                            # if the new belief already exists then we just need to add 
                            # the key of that existing belief to the previous belief

                        else:
                            s1u, _ = self._get_new_state(s, a, unlocked=False)
                            b1u    = b.copy()

                            # check if this belief already exists
                            hip, idxp, check = self._check_belief_exists(btree, [b1u, s1u])
                            # if it doesn't exist then add it to the belief tree
                            # and add its key to the previous belief that gave rise to it
                            if not check:
                                btree[hi][idx]            = [[b1u.copy(), s1u], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[[a, hi, idx]]]
                                idx                      += 1
                            # if the new belief already exists then we just need to add 
                            # the key of that existing belief to the previous belief
                            else:
                                if [a, hip, idxp] not in btree[hi-1][prev_idx][2]:
                                    btree[hi-1][prev_idx][2] += [[[a, hip, idxp]]]

        return btree

    def _get_state_state(self, b, Q):

        '''
        ---
        marginalise T[s, a, s'] over actions with the current policy 

        parameters:
            b -- current belief about transition structure
            Q -- Q values associated with this belief

        returns:
            T -- state-state transition matrix under \pi
        ---
        '''
        Ta     = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s1l, _ = self._get_new_state(s, a, unlocked=False)

                bidx = self._check_uncertain([s, a])
                if bidx is not None:

                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                
                    Ta[s, a, s1u] = b[bidx, 0]/np.sum(b[bidx, :])
                    Ta[s, a, s1l] = b[bidx, 1]/np.sum(b[bidx, :])

                else:
                    Ta[s, a, s1l] = 1

        T = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            qvals = Q[s, :]
            probs = self._policy(qvals)
            for a in range(self.num_actions):
                T[s, :] += probs[a] * Ta[s, a, :]

        return T

    def _build_pneed_tree(self):

        '''
        ---
        Compute Need for each belief state

        ttree -- tree with the estimated probabilities 
        ---
        '''

        # here is the picture:
        #
        #                -
        #               / 
        #              X
        #             / \
        #            /   -
        # A - - - - -
        #            \   -
        #             \ /
        #              -
        #               \
        #                -
        #
        # A is the agent's current state
        # X is the belief at which an update is executed
        # 
        # The path to X is estimated based on 
        # monte-carlo returns in the method called 
        # simulate_trajs()

        ttree = self._simulate_trajs()

        ntree = {hi:{} for hi in range(self.horizon)}

        for hi in range(self.horizon):
            
            for k, vals in self.belief_tree[hi].items():
                
                ntree[hi][k] = 0

                if k not in ttree[hi].keys():
                    continue

                b     = self.belief_tree[hi][k][0][0]
                s     = self.belief_tree[hi][k][0][1]
                Q     = self.belief_tree[hi][k][1].copy()
                T     = self._get_state_state(b, Q)
                SR_k  = np.linalg.inv(np.eye(self.num_states) - self.gamma*T)

                bn      = ttree[hi][k][0]
                bp      = ttree[hi][k][1]

                maskedn = bn[~np.isnan(bn)]
                maskedp = bp[~np.isnan(bn)]
                
                av_SR   = 0
                
                for idx in range(len(maskedn)):
                    
                    SR = SR_k.copy()
                    
                    for i in range(int(maskedp[idx])+1):
                        SR -= (self.gamma**i)*np.linalg.matrix_power(T, i)

                    av_SR += maskedn[idx] + SR[self.state, s]
                
                ntree[hi][k] += av_SR/self.num_sims

        return ntree

    def _imagine_update(self, Q_old, state, b, val, btree):
        
        '''
        ---
        propose a model-based update to a model-free value at a single belief state 

        parameters:
            Q_old -- current MF value
            state -- physical location 
            b     -- belief
            val   -- index
            btree -- belief tree

        returns:
            Q_new -- new updated MF value
        ---
        '''

        q_old_vals = Q_old[state, :].copy()

        tds = []

        # if there are two belief states to which the subject can transition from the 
        # current belief state, it means that this action's outcome is uncertain
        if len(val) == 2:
            
            # retrieve next belief states' indices
            a, hiu, idxu = val[0][0], val[0][1], val[0][2]
            _, hil, idxl = val[1][0], val[1][1], val[1][2]

            # s1u -- new state with unlocked barrier
            s1u          = btree[hiu][idxu][0][1]
            q_prime_u    = btree[hiu][idxu][1][s1u, :].copy()

            # s1l -- new state with locked barrier
            s1l          = btree[hil][idxl][0][1]
            q_prime_l    = btree[hil][idxl][1][s1l, :].copy()

            # reward for new location (old location is just 0)
            y, x = self._convert_state_to_coords(s1u)
            rew  = self.config[y, x]

            # add temporal difference errors for both updates
            tds += [q_old_vals[a] + self.alpha_r*(rew + self.gamma*np.nanmax(q_prime_u) - q_old_vals[a])]
            tds += [q_old_vals[a] + self.alpha_r*(0 + self.gamma*np.nanmax(q_prime_l) - q_old_vals[a])]

        # otherwise there's only one possible next belief state
        else:
            a, hi1, idx1 = val[0][0], val[0][1], val[0][2]
            s1           = btree[hi1][idx1][0][1]
            q_prime      = btree[hi1][idx1][1][s1, :].copy()

            y, x = self._convert_state_to_coords(s1)
            rew  = self.config[y, x]
            tds += [q_old_vals[a] + self.alpha_r*(rew + self.gamma*np.nanmax(q_prime) - q_old_vals[a])]

        # get the new (updated) q value
        Q_new      = Q_old.copy()
        q_new_vals = q_old_vals.copy()

        # perform the update
        if len(tds) != 2: 
            q_new_vals[a] = tds[0]
        else:    
            bidx = self._check_uncertain([state, a])
            b0   = b[bidx, 0]/np.sum(b[bidx, :])
            b1   = 1 - b[bidx, 0]/np.sum(b[bidx, :])
            # weighted by the subject's prior belief 
            q_new_vals[a] = b0*tds[0] + b1*tds[1]

        Q_new[state, :] = q_new_vals

        return Q_new

    def _generate_forward_sequences(self, updates):

        '''
        ---
        generates forward replay sequences 

        parameters:
            updates     -- single-action replay updates to extend 
            pntree      -- need

        retuns:
            seq_updates -- forward sequence updates
        ---
        '''

        seq_updates = []

        # iterate over all single-action updates
        for update in updates:
            for l in range(self.max_seq_len-1):
                
                if l == 0:
                    pool = [update]
                else:
                    pool = deepcopy(tmp)

                tmp  = []
                
                for seq in pool: # take an existing (current) sequence
                    
                    lhi   = seq[0][-1]
                    levb  = seq[-1][-1]

                    if levb < self.xi:
                        continue

                    if lhi == (self.horizon - 2):
                        continue

                    lidx      = seq[1][-1]
                    la        = seq[2][-1, 1]
                    next_idcs = self.belief_tree[lhi][lidx][2]

                    tt = []
                    for next_idx in next_idcs:
                        if len(next_idx) == 2:
                            tt += [next_idx[0]]
                            tt += [next_idx[1]]
                        else:
                            tt += next_idx
                    next_idcs = tt

                    for next_idx in next_idcs:

                        nhi  = next_idx[1]
                        nidx = next_idx[2]

                        if la == next_idx[0]:

                            vals = self.belief_tree[nhi][nidx]
                            s        = vals[0][1]
                            if s in self.goal_states:
                                continue
                            b        = vals[0][0].copy()

                            if s not in seq[2][:, 0]:
                            
                                Q_old  = vals[1].copy()

                                next_next_idcs = vals[2]

                                for next_next_idx in next_next_idcs:
                                    
                                    this_seq = deepcopy(seq)
                                    Q = this_seq[3].copy()

                                    Q_new  = self._imagine_update(Q_old, s, b, next_next_idx, self.belief_tree)

                                    # need   = self.pneed_tree[s]
                                    # gain   = self._compute_gain(Q_old[s, :].copy(), Q_new[s, :].copy())
                                    # evb    = gain * need
                                    a          = next_next_idx[0][0]
                                    gain, evb  = self._generalised_gain(s, a, Q_old, Q_new, self.belief_tree)

                                    if evb >= self.xi:
                                        this_seq[0]  = np.append(this_seq[0], nhi)
                                        this_seq[1]  = np.append(this_seq[1], nidx)
                                        a            = next_next_idx[0][0]
                                        this_seq[2]  = np.vstack((this_seq[2], np.array([s, a])))
                                        Q[s, :]      = Q_new[s, :].copy()
                                        this_seq[3]  = Q.copy()
                                        this_seq[4]  = np.append(this_seq[4], gain.copy())
                                        this_seq[5]  = np.append(this_seq[5], self.pneed_tree.copy())
                                        this_seq[6]  = np.append(this_seq[6], np.sum(this_seq[6])+evb)
                                        tmp         += [deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def _generate_reverse_sequences(self, updates):

        seq_updates = []

        for update in updates:
            for l in range(self.max_seq_len-1):
                
                if l == 0:
                    pool = [update]
                else:
                    pool = deepcopy(tmp)

                tmp  = []
                
                for seq in pool: # take an existing sequence
                    
                    lhi   = seq[0][-1]
                    lidx  = seq[1][-1]
                    levb  = seq[-1][-1]

                    if levb < self.xi:
                        continue

                    for hor in range(self.horizon):

                        for k, vals in self.belief_tree[hor].items():

                            next_idcs = vals[2]

                            if len(next_idcs) == 0:
                                continue

                            for next_idx in next_idcs:

                                if len(next_idx) == 2:
                                    cond = ((next_idx[0][1] == lhi) and (next_idx[0][2] == lidx)) or ((next_idx[1][1] == lhi) and (next_idx[1][2] == lidx))
                                else:
                                    cond = (next_idx[0][1] == lhi) and (next_idx[0][2] == lidx)
                                
                                if cond: # found a prev exp

                                    this_seq = deepcopy(seq)
                                    s        = vals[0][1]
                                    b        = vals[0][0].copy()

                                    if s not in this_seq[2][:, 0]:
                                        
                                        # 
                                        nbtree = deepcopy(self.belief_tree)
                                        Q      = seq[3].copy()
                                        nbtree[lhi][lidx][1] = Q.copy()
                                        Q_old  = nbtree[hor][k][1].copy()

                                        a      = next_idx[0][0]
                                        Q_new  = self._imagine_update(Q_old, s, b, next_idx, nbtree)

                                        # need   = self.pneed_tree[s]
                                        # gain   = self._compute_gain(Q_old[s, :].copy(), Q_new[s, :].copy())
                                        # evb    = gain*need

                                        gain, evb  = self._generalised_gain(s, a, Q_old, Q_new, nbtree)
                                        # gain = self._compute_gain(Q_new[s, :], Q_new[s, :])
                                        # need = self.pneed_tree[hor][k]

                                        # evb  = gain*need
                                        # gain, evb = self._generalised_gain(state, a, pntree, Q_old_this, Q_new_this)
                                        
                                        if evb >= self.xi:
                                            this_seq[0]  = np.append(this_seq[0], hor)
                                            this_seq[1]  = np.append(this_seq[1], k)
                                            a            = next_idx[0][0]
                                            this_seq[2]  = np.vstack((this_seq[2], np.array([s, a])))
                                            Q[s, :]      = Q_new[s, :].copy()
                                            this_seq[3]  = Q.copy()
                                            this_seq[4]  = np.append(this_seq[4], gain.copy())
                                            this_seq[5]  = np.append(this_seq[5], deepcopy(self.pneed_tree))
                                            this_seq[6]  = np.append(this_seq[6], np.sum(this_seq[6])+evb)
                                            tmp         += [deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def _generalised_gain(self, state, a, Q_old_this, Q_new_this, belief_tree):

        gain  = {hi1:{} for hi1 in range(self.horizon)}
        evb   = 0

        for hi1 in range(self.horizon):
            for idx1, vals1 in self.belief_tree[hi1].items():

                gain[hi1][idx1] = 0
                sp              = vals1[0][1]
                bp              = vals1[0][0]
                Q_old           = vals1[1]
                if (sp == state):
                    for val1 in vals1[2]:
                        ap = val1[0][0]
                        if ap == a:
                            Q_new           = self._imagine_update(Q_old, sp, bp, val1, belief_tree)
                            if Q_new[sp, ap] == Q_new_this[sp, a]:
                                need            = self.pneed_tree[hi1][idx1]
                                this_gain       = self._compute_gain(Q_old[sp, :].copy(), Q_new[sp, :].copy())
                                gain[hi1][idx1] = this_gain
                                
                                evb            += need * this_gain

        return gain, evb

    def _generate_single_updates(self):

        '''
        ---
        generate all single-action reply updates

        parameters:
            pntree  -- need tree

        returns:
            updates -- generated single-action replay updates
        ---
        '''

        updates     = []

        for hi in reversed(range(self.horizon-1)):

            # skip to the previous horizon if at the end
            if len(self.belief_tree[hi+1]) == 0:
                continue
            
            # iterate over every belief state
            for idx, vals in self.belief_tree[hi].items():
                
                state = vals[0][1]

                # do not consider if goal state
                if state in self.goal_states:
                    continue
                
                b_this     = vals[0][0] # belief
                Q_old_this = vals[1]    # current Q values

                # iterate over belief states to which it is possible to transition from this belief state
                for val in vals[2]:
                    
                    a          = val[0][0] # action that results in the transition
                    Q_new_this = self._imagine_update(Q_old_this, state, b_this, val, self.belief_tree)     # new updated Q value
                    gain, evb  = self._generalised_gain(state, a, Q_old_this, Q_new_this, self.belief_tree) # gain & evb associated with the update

                    # add to the list of potential updates
                    if evb >= self.xi:
                        updates += [[np.array([hi]), np.array([idx]), np.array([state, a]).reshape(-1, 2), Q_new_this.copy(), [gain], [self.pneed_tree], [evb]]]

        return updates

    def _get_highest_evb(self, updates):
        
        evb_all = np.zeros(len(updates))
        loc     = None
        for idx, upd in enumerate(updates):
            evb = upd[-1][-1]
            evb_all[idx] = evb

        max_evb = np.max(evb_all)
        loc     = np.argwhere(evb_all == max_evb)[0]

        if len(loc) > 1:
            tmp_updates = updates[loc]
            min_len     = self.max_seq_len
            for idx, upd in enumerate(tmp_updates):
                if len(upd[-1]) <= min_len:
                    this_loc = loc[idx]
                    min_len  = len(upd[-1])
            loc = this_loc

        return loc[0], max_evb

    def _replay(self):
        
        gain_history = [None]
        need_history = [None]

        self.belief_tree = self._build_belief_tree()
        self.pneed_tree  = self._build_pneed_tree()
        
        Q_history        = [deepcopy(self.belief_tree)]

        num = 1
        while True:
            updates = self._generate_single_updates()

            if len(updates) == 0:
                break 
            
            if self.sequences:
                fwd_updates  = self._generate_forward_sequences(updates)
                rev_updates  = self._generate_reverse_sequences(updates)
                if len(rev_updates) > 0:
                    updates += rev_updates
                if len(fwd_updates) > 0:
                    updates += fwd_updates

            idx, evb = self._get_highest_evb(updates)

            if idx is None:
                break

            if evb < self.xi:
                break
            else:

                hi  = updates[idx][0]
                k   = updates[idx][1]
                
                s  = updates[idx][2][:, 0]
                a  = updates[idx][2][:, 1]
                
                Q_new = updates[idx][3].copy()

                gain  = updates[idx][4]
                need  = updates[idx][5]

                for sidx, si in enumerate(s):
                    Q_old = self.belief_tree[hi[sidx]][k[sidx]][1].copy()
                    b     = self.belief_tree[hi[sidx]][k[sidx]][0][0]
                    print('%u - Replay %u/%u [<%u>, %u] horizon %u, q_old: %.2f, q_new: %.2f, evb: %.2f'%(num, sidx+1, len(s), si, a[sidx], hi[sidx], Q_old[si, a[sidx]], Q_new[si, a[sidx]], evb), flush=True)
                    print(b)
                    print('---')
                    Q_old[si, a[sidx]] = Q_new[si, a[sidx]].copy()
                    self.belief_tree[hi[sidx]][k[sidx]][1] = Q_old.copy()
            
                    if np.array_equal(b, self.M):
                        self.Q[si, a[sidx]] = Q_new[si, a[sidx]].copy()
                    
                    Q_history += [deepcopy(self.belief_tree)]

                need_history  += [need]
                gain_history  += [gain]

                self.pneed_tree = self._build_pneed_tree()

                num += 1

        return Q_history, gain_history, need_history

    def run_simulation(self, num_steps=100, save_path=None):

        '''
        ---
        Main loop for the simulation

        num_steps    -- number of simulation steps
        start_replay -- after which step to start replay
        reset_pior   -- whether to reset transition prior before first replay bout
        save_path    -- path for saving data after replay starts
        ---
        '''

        if save_path:
            self.save_path = save_path

            if os.path.isdir(self.save_path):
                shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)
        else:
            self.save_path = None

        replay = False
        num_replay_moves = 0

        for move in range(num_steps):
            
            s      = self.state

            # choose action and receive feedback
            probs  = self._policy(self.Q[s, :])
            a      = np.random.choice(range(self.num_actions), p=probs)

            bidx = self._check_uncertain([s, a])
            if bidx is not None:
                if self.barriers[bidx]:
                    s1, r  = self._get_new_state(s, a, unlocked=False)
                else:
                    s1, r  = self._get_new_state(s, a, unlocked=True)
                
                # update belief
                self.M = self._belief_plan_update(self.M, bidx, s, s1)

                if replay:
                    # fetch Q values of the new belief
                    for hi in range(self.horizon):
                        for k, vals in self.belief_tree[hi].items():
                            b = vals[0][0]
                            if np.array_equal(self.M, b):
                                state  = vals[0][1]
                                Q_vals = self.belief_tree[hi][k][1]
                                self.Q[state, :] = Q_vals[state, :].copy()

            else:
                s1, r  = self._get_new_state(s, a, unlocked=True)

            q_old  = self.Q[s, a]

            # update MF Q values
            self._qval_update(s, a, r, s1)

            print('Move %u/%u, [<%u, [%.2f, %.2f, %.2f]>, %u], q_old: %.2f, q_new: %.2f\n'%(move, num_steps, s, self.M[0, 0]/self.M[0, :].sum(), self.M[1, 0]/self.M[1, :].sum(), self.M[2, 0]/self.M[2, :].sum(), a, q_old, self.Q[s, a]), flush=True)

            # transition to new state
            self.state = s1

            if self.state in self.goal_states:
                replay = True

            if replay:
                Q_history, gain_history, need_history = self._replay()
                num_replay_moves += 1
                if num_replay_moves >= 20:
                    return None

            if save_path:
                if replay:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=Q_history, M=self.M, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])
                else:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=self.Q, M=self.M, move=[s, a, r, s1])

            if s1 in self.goal_states:
                self.state = self.start_state

        return None


class AgentMDP(PanAgent, Environment):

    def __init__(self, *configs):
        
        '''
        ----
        configs is a list containing 
                    [0] panagent parameters; [1] agent parameters; [2] environment parameters
        ----
        '''
        
        pag_config = configs[0]
        ag_config  = configs[1]
        env_config = configs[2]
        
        Environment.__init__(self, **env_config)
        PanAgent.__init__(self, **pag_config)
        self.__dict__.update(**ag_config)

        self.state = self.start_state

        # initialise everything
        self._init_q_values()
        self._init_t_model()
        self._init_replay_buff()

        return None

    def _normalise_t_model(self):

        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.T[s, a, :] /= np.sum(self.T[s, a, :])
        
        return None

    def _init_t_model(self):

        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s1, _ = self._get_new_state(s, a, unlocked=True)
                self.T[s, a, s1] += 1
        
        self._normalise_t_model()
        
        return None

    def _update_t_model(self, s, a, s1):

        # state_vec     = np.zeros(self.num_states)
        # state_vec[s1] = 1

        # self.T[s, a, :] = self.T[s, a, :] + (state_vec - self.T[s, a, :])

        self.T[s, a, :]  = np.zeros(self.num_states)
        self.T[s, a, s1] = 1

        self._normalise_t_model()

        return None

    def _init_replay_buff(self):

        self.M = np.full((self.num_states*self.num_actions, 2), np.nan)

        return None

    def _update_replay_buff(self, s, a, r, s1):

        self.M[s*self.num_actions + a, 0] = r
        self.M[s*self.num_actions + a, 1] = s1

        return None

    def _replay(self):
        
        Q_history    = [self.Q.copy()]
        backups      = [None]

        while True:

            Q_new = self.Q_nans.copy()
            gain  = Q_new.copy()
            evb   = Q_new.copy()
            SR    = self._compute_need(self.T, self.Q, inv_temp=self.beta)
            need  = SR[self.state, :]

            for s in np.delete(range(self.num_states), self.goal_states + self.nan_states):
                q_old = self.Q[s, :].copy()
                for a in range(self.num_actions):
                    if ~np.isnan(self.Q[s, a]) and ~np.isnan(self.M[s*self.num_actions+a, 1]):
                        q_new       = q_old.copy()
                        rr          = int(self.M[s*self.num_actions + a, 0])
                        s1r         = int(self.M[s*self.num_actions + a, 1])
                        q_new[a]   += self.alpha_r*(rr + self.gamma*np.nanmax(self.Q[s1r, :]) - self.Q[s, a])
                        
                        Q_new[s, a] = q_new[a]

                        gain[s, a]  = self._compute_gain(q_old, q_new, inv_temp=self.beta)
                        evb[s, a]   = gain[s, a] * need[s]
            
            if len(backups) == 1:
                gain_history   = [gain.copy()]
                need_history   = [need.copy()]
            else:
                gain_history  += [gain.copy()]
                need_history  += [need.copy()]

            max_evb = np.nanmax(evb[:])
            if max_evb >= self.xi:
                evb_idx        = np.argwhere(evb == max_evb)
                sr, ar         = evb_idx[0, :]
                self.Q[sr, ar] = Q_new[sr, ar]

                backups       += [[sr, ar]]
                Q_history     += [self.Q.copy()]

            else:
                return Q_history, gain_history, need_history, backups
                    
    def run_simulation(self, num_steps=100, save_path=None):

        '''
        ---
        Main loop for the simulation

        num_steps    -- number of simulation steps
        start_replay -- after which step to start replay
        reset_pior   -- whether to reset transition prior before first replay bout
        save_path    -- path for saving data after replay starts
        ---
        '''

        if save_path:
            self.save_path = save_path

            if os.path.isdir(self.save_path):
                shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)
        else:
            self.save_path = None

        # replay = False

        for move in range(num_steps):
            
            if move >= 3000:
                self.barriers = [0, 1, 0]
            else:
                self.barriers = [1, 1, 0]

            s      = self.state

            # choose action and receive feedback
            probs  = self._policy(self.Q[s, :])
            a      = np.random.choice(range(self.num_actions), p=probs)

            bidx   = self._check_blocked([s, a])
            if bidx is not None:
                if not self.barriers[bidx]:
                    locked = True
                else:
                    locked = False
            else:
                locked = False

            s1, r  = self._get_new_state(s, a, unlocked=locked)

            # update MF Q values
            self._qval_update(s, a, r, s1)

            # update replay buffer
            self._update_replay_buff(s, a, r, s1)

            # update transition model
            self._update_t_model(s, a, s1)

            # if (s1 == self.goal_state) or (self.state == self.start_state):
                # replay = True

            self.state = s1

            # if replay:
            Q_history, gain_history, need_history, backups = self._replay()

            if save_path:
                # if replay:
                np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=Q_history, replays=backups, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])
                # else:
                    # np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=self.Q, move=[s, a, r, s1])

            if self.state in self.goal_states:
                self.state = self.start_state

            self._mf_forget()
            # replay = False

        return None