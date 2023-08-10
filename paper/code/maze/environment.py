import numpy as np

class Environment():

    def __init__(self, **p):

        '''
        ----
        config                -- matrix which specifies the env
        blocked_state_actions -- list with state-action pairs [s, a] which are blocked
        start_coords          -- start state coords
        goal_coords           -- goal state coords
        ----
        '''

        self.__dict__.update(**p)

        self.num_states  = self.num_x_states*self.num_y_states
        self.start_state = self._convert_coords_to_state(self.start_coords)
        
        self._init_reward()

        return None

    def _init_reward(self):

        self.config      = np.zeros((self.num_y_states, self.num_x_states))
        self.goal_states = []
        for idx, goal in enumerate(self.goal_coords):
            self.config[goal[0], goal[1]] = self.rew_value[idx]
            self.goal_states += [self._convert_coords_to_state(goal)]

    def _get_new_state(self, s, a, unlocked=False):

        '''
        ----
        s        -- current state of the agent
        a        -- chosen action
        unlocked -- whether the action is available or not (for blocked_state_actions)
        ----
        '''

        if s in self.goal_states:
            return self.start_state, self.config.ravel()[s]

        if (self._check_blocked([s, a]) is not None) and (unlocked == False):
            return s, self.config.ravel()[s]

        if s in self.nan_states:
            return s, 0

        y_coord, x_coord = self._convert_state_to_coords(s)

        # ----
        # first consider edge cases
        # at the top and choose up
        case1 = (y_coord == 0) and (a == 0)
        # at the bottom and choose down
        case2 = (y_coord == self.num_y_states - 1) and (a == 1)
        # at the left edge and choose left
        case3 = (x_coord == 0) and (a == 2)
        # at the right edge and choose right
        case4 = (x_coord == self.num_x_states - 1) and (a == 3)

        if case1 or case2 or case3 or case4:
            r = self.config[y_coord, x_coord]
            return s, r
        else:

            # ----
            # choose up
            if a == 0:
                x1_coord, y1_coord = x_coord, y_coord - 1
            # choose down
            elif a == 1:
                x1_coord, y1_coord = x_coord, y_coord + 1
            # choose left 
            elif a == 2:
                x1_coord, y1_coord = x_coord - 1, y_coord
            # choose right
            else:
                x1_coord, y1_coord = x_coord + 1, y_coord

            s1 = self._convert_coords_to_state([y1_coord, x1_coord])

            if s1 in self.nan_states:
                r  = self.config[y_coord, x_coord]
                return s, r

            # check barriers
            if (unlocked == True) or self._check_uncertain([s, a]) is None:
                r  = self.config[y1_coord, x1_coord]
                return s1, r
            else:
                r = self.config[y_coord, x_coord]
                return s, r
            
    def _convert_state_to_coords(self, s):

        y_coord = s // self.num_x_states
        x_coord = s % self.num_x_states

        return [y_coord, x_coord]

    def _convert_coords_to_state(self, coords: list):

        y_coord, x_coord = coords
        states = np.arange(self.num_states).reshape(self.num_y_states, self.num_x_states)

        return states[y_coord, x_coord]

    # def _init_barriers(self, bars=None):

    #     if bars is None:
    #         return None
    #     else:
    #         self.barriers = bars

    #     return None

    def _init_q_values(self):

        self.Q = np.zeros((self.num_states, self.num_actions))

        # set edge Q values to np.nan
        for s in np.delete(range(self.num_states), self.goal_states + self.nan_states):
            for a in range(self.num_actions):
                # bidx = self._check_blocked([s, a])
                # if bidx is None:
                s1, _ = self._get_new_state(s, a, unlocked=True)
                if self._check_blocked([s, a]) is None and self._check_uncertain([s, a]) is None:
                    if (s1 == s):
                        self.Q[s, a] = np.nan

        if len(self.nan_states) > 0:
            for s in self.nan_states:
                self.Q[s, :] = np.nan

        # if self.env_name == 'tolman1':
        #     self.Q[8,  1] = np.nan
        # elif self.env_name == 'tolman2':
        #     self.Q[20, 1] = np.nan
        # elif self.env_name == 'tolman3':
        #     self.Q[8,  3] = np.nan
        # elif self.env_name == 'u':
        #     self.Q[1,  2] = np.nan
        if self.env_name == 'tolman123_nocheat':
            self.Q[8,  1] = np.nan
            self.Q[20, 1] = np.nan
        elif self.env_name == 'tolman1234':
            self.Q[8, 1]  = np.nan
            self.Q[20, 2] = np.nan
            self.Q[32, 2] = np.nan
        else: pass

        self.Q_nans = self.Q.copy()

        return None

    def _solve_mb(self, eps, barriers=None):
        
        if barriers is None:
            barriers = self.barriers
        else:
            self.barriers = barriers

        Q_MB  = self.Q_nans.copy()
        delta = 1
        while delta > eps:
            Q_MB_new = Q_MB.copy()
            for s in np.delete(range(self.num_states), self.goal_states + self.nan_states):
                for a in range(self.num_actions):
                    if ~np.isnan(Q_MB[s, a]):
                        bidx = self._check_blocked([s, a])
                        if bidx is not None:
                            if self.barriers[bidx]:
                                s1, r = self._get_new_state(s, a, unlocked=False)
                            else:
                                s1, r = self._get_new_state(s, a, unlocked=True)
                        else:
                            s1, r = self._get_new_state(s, a, unlocked=True)
                        Q_MB_new[s, a] += r + self.gamma*np.nanmax(Q_MB[s1, :]) - Q_MB_new[s, a]
            diff  = np.abs(Q_MB_new - Q_MB)
            delta = np.nanmax(diff[:])
            Q_MB  = Q_MB_new

        return Q_MB_new

    def _check_uncertain(self, sa: list):

        for bidx, l in enumerate(self.uncertain_states_actions):
            if sa in l:
                return bidx
        
        return None

    def _check_blocked(self, sa: list):

        for bidx, l in enumerate(self.blocked_state_actions):
            if sa in l:
                return bidx
        
        return None