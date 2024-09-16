import numpy as np
import copy

class Tree:

    def __init__(self, **p):
        
        '''
        ----
        MAB agent

        root_belief   -- current posterior belief
        root_q_values -- MF Q values at the current state
        policy_temp   -- inverse temperature
        ----
        '''
        
        self.__dict__.update(**p)

        if self.sequences:
            if self.max_seq_len is None:
                self.max_seq_len = self.horizon-1

        self.belief_tree = self._build_belief_tree()
        qval_tree        = self._build_qval_tree()
        if self.rand_init:
            self.qval_tree = self._adjust_qval_tree(qval_tree)
        else:
            self.qval_tree = qval_tree

        return None

    def _policy(self, q_values):

        '''
        ----
        Agent's policy

        q_values -- q values at the current state
        temp     -- inverse temperature
        type     -- softmax / greeedy
        ----
        '''

        if np.all(q_values == 0):
            return np.array([0.5, 0.5])

        if self.beta:
            t = self.beta
        else:
            t = 1
            
        if t != 'greedy':
            return np.exp(q_values*t)/np.sum(np.exp(q_values*t))
        elif t == 'greedy':
            if np.all(q_values == q_values.max()):
                a = np.random.choice([1, 0], p=[0.5, 0.5])
                return np.array([a, 1-a])
            else:
                return np.array(q_values >= q_values.max()).astype(int)
        else:
            raise KeyError('Unknown policy type')

    def _value(self, qvals):

        return np.dot(self._policy(qvals), qvals)

    def evaluate_policy(self, qval_tree):

        '''
        ----
        Evaluate the tree policy

        qval_tree -- tree with Q values for each belief
        ----
        '''

        eval_tree  = {hi:{} for hi in range(self.horizon)}

        # then propagate those values backwards
        for hi in reversed(range(self.horizon-1)):
            for idx, vals in self.belief_tree[hi].items():

                b     = vals[0]

                eval_tree[hi][idx] = 0

                qvals = qval_tree[hi][idx].copy()
                probs = self._policy(qvals)

                next_idcs = vals[1]

                for next_idx in next_idcs:
                    
                    a    = next_idx[0]
                    
                    if self.arms[a] == 'known':

                        idx1 = next_idx[1][0]

                        if hi == self.horizon - 2:
                            v_primes = [self._value(qval_tree[hi+1][idx1]), self._value(qval_tree[hi+1][idx1])]
                        else:
                            v_primes = [eval_tree[hi+1][idx1], eval_tree[hi+1][idx1]]

                        b0 = b[a]

                    elif self.arms[a] == 'unknown':

                        idx1 = next_idx[1][0]
                        idx2 = next_idx[1][1]

                        if hi == self.horizon - 2:
                            v_primes = [self._value(qval_tree[hi+1][idx1]), self._value(qval_tree[hi+1][idx2])]
                        else:
                            v_primes = [eval_tree[hi+1][idx1], eval_tree[hi+1][idx2]]

                        b0 = b[a][0]/np.sum(b[a][:])

                    b1 = 1 - b0

                    eval_tree[hi][idx] += probs[a] * (b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1]))

        return eval_tree[0][0]

    def _belief_update(self, curr_belief, arm, rew):

        '''
        ----
        Bayesian belief updates for beta prior

        curr_belief -- matrix with the current beliefs
        arm         -- chosen arm 
        rew         -- received reward
        ----
        ''' 

        b_next = copy.deepcopy(curr_belief)

        if self.arms[arm] == 'known':
            return b_next

        if rew == 1:
            b_next[arm][0] += 1
        else:
            b_next[arm][1] += 1
        return b_next

    def _build_belief_tree(self):

        '''
        ----
        Generate planning belief tree

        h -- horizon
        ----
        '''

        # initialise the hyperstate tree
        belief_tree = {hi:{} for hi in range(self.horizon)}
        
        idx  = 0
        belief_tree[0][idx] = [self.root_belief, []]

        for hi in range(1, self.horizon):
            
            idx = 0

            for prev_idx, vals in belief_tree[hi-1].items():

                b = vals[0]

                for a in range(2):
                    
                    if self.arms[a] == 'known':
                        # generates 1 outcome
                        # same belief state since no learning
                        b1n  = copy.deepcopy(b)
                        # still add this belief state to the next horizon
                        belief_tree[hi][idx] = [b1n, []]
                        # add its idx to belief state which generated it
                        belief_tree[hi-1][prev_idx][-1] += [[a, [idx]]]
                        # increment idx
                        idx += 2

                    elif self.arms[a] == 'unknown':
                        # generates 2 outcomes

                        # success
                        r    = 1
                        b1s  = self._belief_update(b, a, r)
                        belief_tree[hi][idx] = [b1s, []]

                        # fail
                        r    = 0
                        b1f  = self._belief_update(b, a, r)
                        belief_tree[hi][idx+1] = [b1f, []]
                        
                        belief_tree[hi-1][prev_idx][-1] += [[a, [idx, idx+1]]]

                        idx += 2

        return belief_tree

    def full_updates(self):
        '''
        Compute full Bayes-optimal Q values at the root 
        (up to the specified horizon)
        ----
        tree  -- belief tree
        gamma -- discount factor
        ----
        '''

        qval_tree = self._build_qval_tree()

        # then propagate those values backwards
        for hi in reversed(range(self.horizon-1)):
            for idx, vals in self.belief_tree[hi].items():
                
                b  = vals[0]
                
                qval_tree[hi][idx] = np.zeros(2)

                next_idcs = vals[1]
                
                for next_idx in next_idcs:
                    a    = next_idx[0]

                    if self.arms[a] == 'known':
                        
                        idx1 = next_idx[1][0]

                        b0 = b[a]

                        v_primes = [np.max(qval_tree[hi+1][idx1]), np.max(qval_tree[hi+1][idx1])]

                    elif self.arms[a] == 'unknown':

                        # generates 2 outcomes
                        idx1 = next_idx[1][0]
                        idx2 = next_idx[1][1]

                        b0 = b[a][0]/np.sum(b[a][:])
                
                        v_primes = [np.max(qval_tree[hi+1][idx1]), np.max(qval_tree[hi+1][idx2])]

                    b1 = 1 - b0

                    qval_tree[hi][idx][a] = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

        return qval_tree

    def _build_qval_tree(self):
        
        qval_tree = {hi:{} for hi in range(self.horizon)}

        # set the leaf values as certainty-equivalent reward
        for hi in range(self.horizon):
            for idx, vals in self.belief_tree[hi].items():
                if (hi == self.horizon - 1):
                    b  = vals[0]

                    if self.arms[0] == 'known':
                        b0 = b[0]
                    elif self.arms[0] == 'unknown':
                        b0 = b[0][0]/np.sum(b[0])
                    


                    if self.arms[1] == 'known':
                        b1 = b[1]
                    elif self.arms[1] == 'unknown':
                        b1 = b[1][0]/np.sum(b[1])

                    b0 = b0/(1-self.gamma)
                    b1 = b1/(1-self.gamma)

                else:
                    b0 = 0.0
                    b1 = 0.0
                qval_tree[hi][idx] = np.array([b0, b1])

        return qval_tree

    def _build_empty_tree(self):

        empty_tree = {hi:{} for hi in range(self.horizon)}

        for hi in range(self.horizon):
            for k in self.belief_tree[hi].keys():

                empty_tree[hi][k] = np.array([0.0, 0.0])

        return empty_tree

    def _adjust_qval_tree(self, qval_tree):

        full_qval_tree = self.full_updates()

        all_hors  = []
        all_idcs  = []
        for hi in range(self.horizon - 1):
            num_idcs  = len(self.belief_tree[hi].keys())
            all_idcs += [i for i in self.belief_tree[hi].keys()]
            all_hors += [hi]*num_idcs

        for hi in range(self.horizon - 1):
            for idx in full_qval_tree[hi].keys():  
                ch = np.random.choice(range(len(all_hors)))
                qval_tree[hi][idx] = full_qval_tree[all_hors[ch]][all_idcs[ch]]
                all_hors.pop(ch)
                all_idcs.pop(ch)

        return qval_tree

    def _build_need_tree(self):

        need_tree  = {hi:{} for hi in range(self.horizon)}

        # need at current belief state is 1
        need_tree[0][0] = 1

        for hi in range(1, self.horizon):
            for prev_idx, vals in self.belief_tree[hi-1].items():
                # compute Need with the behavioural (softmax) policy
                prev_need    = need_tree[hi-1][prev_idx]
                policy_proba = self._policy(self.qval_tree[hi-1][prev_idx])
                
                next_idcs = vals[1]
                for next_idx in next_idcs:
                    a    = next_idx[0]
                    
                    b  = vals[0]

                    if self.arms[a] == 'known':
                        b0 = b[a]
                        b1 = 1 - b0

                        idx1 = next_idx[1][0]
                        need_tree[hi][idx1] = policy_proba[a]*prev_need*self.gamma

                    elif self.arms[a] == 'unknown':
                        b0 = b[a][0]/np.sum(b[a][:])
                        b1 = 1 - b0

                        idx1 = next_idx[1][0]
                        need_tree[hi][idx1] = policy_proba[a]*b0*prev_need*self.gamma

                        idx2 = next_idx[1][1]
                        need_tree[hi][idx2] = policy_proba[a]*b1*prev_need*self.gamma

        return need_tree

    def _get_highest_evb(self, updates):
        
        max_evb = 0
        idx     = None
        for uidx, update in enumerate(updates):
            evb = update[-1][-1]
            if evb > max_evb:
                max_evb = evb
                idx     = uidx

        if max_evb > self.xi:
            return idx, max_evb
        else:
            return None, None

    def _find_belief(self, hi, z):

        same_beliefs = []

        for idx, vals in self.belief_tree[hi].items():

                b = vals[0]

                if np.array_equal(np.array(b[0]), np.array(z[0])) and np.array_equal(np.array(b[1]), np.array(z[1])):

                    same_beliefs += [[hi, idx]]
                
        return same_beliefs

    def _generate_single_updates(self):

        updates = []

        self.gain_tree = self._build_empty_tree()

        for hi in reversed(range(self.horizon-1)):
            for idx, vals in self.belief_tree[hi].items():
                
                q    = self.qval_tree[hi][idx].copy() # current Q values of this belief state
                b    = vals[0]
                # need = self.need_tree[hi][idx]

                # compute the new (updated) Q value 
                next_idcs = vals[1]
                for next_idx in next_idcs:
                    
                    a    = next_idx[0]

                    if self.arms[a] == 'known':
                        b0 = b[a]

                        idx1 = next_idx[1][0]

                        v_primes = [np.max(self.qval_tree[hi+1][idx1]), np.max(self.qval_tree[hi+1][idx1])] # values of next belief states

                    elif self.arms[a] == 'unknown':
                        b0 = b[a][0]/np.sum(b[a][:])

                        idx1 = next_idx[1][0]
                        idx2 = next_idx[1][1]

                        v_primes = [np.max(self.qval_tree[hi+1][idx1]), np.max(self.qval_tree[hi+1][idx2])] # values of next belief states

                    b1 = 1 - b0

                    q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

                    if a == 0:
                        q_new = np.array([q_upd, q[1]])
                    else:
                        q_new = np.array([q[0], q_upd])
                    
                    # same_beliefs = self._find_belief(hi, b)
                    same_beliefs = [[hi, idx]]

                    probs_before = self._policy(q)
                    probs_after  = self._policy(q_new)
                    gain         = np.dot(probs_after-probs_before, q_new)
                    # need         = self.need_tree[hi][idx]
                    # evb          = gain * need

                    his, idcs, ays, gains, needs, evbs = [], [], [], [], [], []
                    q_news = np.tile(q_new, (len(same_beliefs), 1))

                    for vs in same_beliefs:

                        his   += [vs[0]]
                        idcs  += [vs[1]]
                        ays   += [a]
                        gains += [gain]
                        needs += [self.need_tree[vs[0]][vs[1]]]
                        evbs  += [needs[-1]*gains[-1]]
                    
                    updates += [[np.array(his), np.array(idcs), np.array(ays), q_news.copy(), np.array(gains), np.array(needs), np.array(np.cumsum(evbs))]]
                    self.gain_tree[hi][idx][a] = gain
                    # updates += [[np.array(hi), np.array(idx), np.array(a), q_new.copy(), np.array(gain), np.array(need), np.array(evb)]]

        return updates

    def _generate_forward_sequences(self, updates):

        seq_updates = []

        for update in updates:
        
            for l in range(self.max_seq_len - 1):

                if l == 0:
                    pool = [copy.deepcopy(update)]
                else:
                    pool = copy.deepcopy(tmp)

                tmp = []

                for seq in pool:

                    prev_hi  = seq[0][-1] # horizon of the previous update

                    if (prev_hi == self.horizon-2):
                        break 
                    
                    prev_idx = seq[1][-1] # idx of the previous belief
                    prev_a   = seq[2][-1] # previous action

                    # belief idcs from which we consider adding an action
                    prev_next_idcs = self.belief_tree[prev_hi][prev_idx][1]

                    for prev_next_idx in prev_next_idcs:
                        
                        if len(prev_next_idx) == 0:
                            break
                        
                        if prev_next_idx[0] == prev_a:
                            prev_idx1 = prev_next_idx[1][0]
                            prev_idx2 = prev_next_idx[1][1]

                            for idx in [prev_idx1, prev_idx2]:
                                
                                b = self.belief_tree[prev_hi+1][idx][0]
                                q = self.qval_tree[prev_hi+1][idx].copy()

                                next_idcs = self.belief_tree[prev_hi+1][idx][1]
                            
                                for next_idx in next_idcs:

                                    a    = next_idx[0]
                                    idx1 = next_idx[1][0]
                                    idx2 = next_idx[1][1]

                                    v_primes = [np.max(self.qval_tree[prev_hi+2][idx1]), np.max(self.qval_tree[prev_hi+2][idx2])] # values of next belief states

                                    # new (updated) Q value for action [a]
                                    if self.arms[a] == 'known':
                                        b0 = b[a, 0]
                                        b1 = 1 - b[a, 0]
                                    
                                    elif self.arms[a] == 'unknown':
                                        b0 = b[a, 0]/np.sum(b[a, :])
                                        b1 = b[a, 1]/np.sum(b[a, :])

                                    q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

                                    if a == 0:
                                        q_new = np.array([q_upd, q[1]])
                                    else:
                                        q_new = np.array([q[0], q_upd])
                                    
                                    probs_before = self._policy(q)
                                    probs_after  = self._policy(q_new)
                                    need         = self.need_tree[prev_hi+1][idx]
                                    gain         = np.dot(probs_after-probs_before, q_new)
                                    evb          = gain*need

                                    if self.constrain_seqs:
                                        if evb > self.xi:
                                            this_seq     = copy.deepcopy(seq)
                                            this_seq[0]  = np.append(this_seq[0], prev_hi+1)
                                            this_seq[1]  = np.append(this_seq[1], idx)
                                            this_seq[2]  = np.append(this_seq[2], a)
                                            this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                            this_seq[4]  = np.append(this_seq[4], gain)
                                            this_seq[5]  = np.append(this_seq[5], need)
                                            this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                            tmp += [copy.deepcopy(this_seq)]
                                    else:
                                        this_seq     = copy.deepcopy(seq)
                                        this_seq[0]  = np.append(this_seq[0], prev_hi+1)
                                        this_seq[1]  = np.append(this_seq[1], idx)
                                        this_seq[2]  = np.append(this_seq[2], a)
                                        this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                        this_seq[4]  = np.append(this_seq[4], gain)
                                        this_seq[5]  = np.append(this_seq[5], need)
                                        this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                        tmp += [copy.deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def _generate_reverse_sequences(self, updates):

        seq_updates = []

        for update in updates:
        
            for l in range(self.max_seq_len - 1):

                if l == 0:
                    pool = [copy.deepcopy(update)]
                else:
                    pool = copy.deepcopy(tmp)

                tmp = []

                for seq in pool:

                    prev_hi  = seq[0][-1]

                    if (prev_hi == 0):
                        break 

                    prev_idx = seq[1][-1]
                    q_seq    = seq[3][-1, :].copy()

                    # find previous belief
                    for idx, vals in self.belief_tree[prev_hi-1].items():

                        next_idcs = vals[1]

                        for next_idx in next_idcs:

                            if (next_idx[1][0] == prev_idx) or (next_idx[1][1] == prev_idx):
                        
                                qval_tree = copy.deepcopy(self.qval_tree)
                                qval_tree[prev_hi][prev_idx] = q_seq.copy()

                                q    = self.qval_tree[prev_hi-1][idx]
                                b    = vals[0]

                                a    = next_idx[0]
                                idx1 = next_idx[1][0]
                                idx2 = next_idx[1][1]

                                v_primes = [np.max(qval_tree[prev_hi][idx1]), np.max(qval_tree[prev_hi][idx2])] # values of next belief states

                                # new (updated) Q value for action [a]
                                if self.arms[a] == 'known':
                                    b0 = b[a, 0]
                                    b1 = 1 - b[a, 0]
                                
                                elif self.arms[a] == 'unknown':
                                    b0 = b[a, 0]/np.sum(b[a, :])
                                    b1 = b[a, 1]/np.sum(b[a, :])

                                q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

                                if a == 0:
                                    q_new = np.array([q_upd, q[1]])
                                else:
                                    q_new = np.array([q[0], q_upd])
                                
                                probs_before = self._policy(q)
                                probs_after  = self._policy(q_new)
                                need         = self.need_tree[prev_hi-1][idx]
                                gain         = np.dot(probs_after-probs_before, q_new)
                                evb          = gain*need

                                if self.constrain_seqs:
                                    if evb > self.xi:
                                        this_seq     = copy.deepcopy(seq)
                                        this_seq[0]  = np.append(this_seq[0], prev_hi-1)
                                        this_seq[1]  = np.append(this_seq[1], idx)
                                        this_seq[2]  = np.append(this_seq[2], a)
                                        this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                        this_seq[4]  = np.append(this_seq[4], gain)
                                        this_seq[5]  = np.append(this_seq[5], need)
                                        this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                        tmp += [copy.deepcopy(this_seq)]
                                else:
                                    this_seq     = copy.deepcopy(seq)
                                    this_seq[0]  = np.append(this_seq[0], prev_hi-1)
                                    this_seq[1]  = np.append(this_seq[1], idx)
                                    this_seq[2]  = np.append(this_seq[2], a)
                                    this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                    this_seq[4]  = np.append(this_seq[4], gain)
                                    this_seq[5]  = np.append(this_seq[5], need)
                                    this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                    tmp += [copy.deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def replay_updates(self):
        '''
        Perform replay updates in the belief tree
        '''
        self.need_tree = self._build_need_tree()
        self.gain_tree = self._build_empty_tree()

        backups      = [None]
        qval_history = [copy.deepcopy(self.qval_tree)]
        need_history = [copy.deepcopy(self.need_tree)]
        gain_history = [copy.deepcopy(self.gain_tree)]

        # compute evb for every backup
        num = 1
        while True:

            updates = self._generate_single_updates()
            
            # generate sequences
            if self.sequences:
                
                fwd_seq_updates = self._generate_forward_sequences(updates)
                rev_seq_updates = self._generate_reverse_sequences(updates)

                if len(fwd_seq_updates) > 0:
                    updates += fwd_seq_updates
                if len(rev_seq_updates) > 0:
                    updates += rev_seq_updates

            uidx, evb = self._get_highest_evb(updates)

            if uidx is None:
                return qval_history, need_history, gain_history, backups
            
            # execute update (replay) with the highest evb
            update = updates[uidx]
            his    = update[0]
            idcs   = update[1]
            aas    = update[2]
            q_news = update[3]
            gains  = update[4]
            needs  = update[5]
            evbs   = update[6]

            for idx, hi in enumerate(his):
                
                q_old = self.qval_tree[hi][idcs[idx]]
                q_new = q_news[idx, :].copy()
                self.qval_tree[hi][idcs[idx]] = q_new.copy()
                print('%u -- Replay %u/%u -- [%u, %u, %u], q_old: %.2f, q_new: %.2f, gain: %.3f, need: %.3f, evb: %.3f'%(num, idx+1, len(his), hi, idcs[idx], aas[idx], q_old[aas[idx]], q_new[aas[idx]], gains[idx], needs[idx], evbs[idx]))

            # save history
            self.need_tree = self._build_need_tree()
            
            qval_history += [copy.deepcopy(self.qval_tree)]
            need_history += [copy.deepcopy(self.need_tree)]
            gain_history += [copy.deepcopy(self.gain_tree)]
            backups      += [[his, idcs, aas]]

            num += 1