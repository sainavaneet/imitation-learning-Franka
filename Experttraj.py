import numpy as np 

class ExpertTraj: 

    def __init__(self, env_name): 

        self.exp_states = np.loadtxt("save_data_state2.txt") 

        self.exp_actions = np.loadtxt("save_data_action2.txt") 

        self.n_transitions = len(self.exp_actions) 

    def sample(self, batch_size): 

        indexes = np.random.randint(0, self.n_transitions, size=batch_size) 

        state, action = [], [] 

        for i in indexes: 

            s = self.exp_states[i] 

            a = self.exp_actions[i] 

            state.append(np.array(s, copy=False)) 

            action.append(np.array(a, copy=False)) 

        return np.array(state), np.array(action) 
