import torch
import numpy as np
from collections import defaultdict


class SwarmMemory(object):
    def __init__(self, num_agents=2, action_dim=2, max_size=int(1e5)):
        self.num_agents = num_agents
        self.memory = [Memory(action_dim=action_dim, max_size=max_size) for _ in range(num_agents)]

    @staticmethod
    # TODO WATCH THIS!! DOES THIS EVEN WORK AS INTENDED???
    def get_agent_state(state, agent_id):
        if isinstance(state, tuple):
            tuple_state = tuple()
            for state_ in state:
                if isinstance(state_[agent_id], np.ndarray):
                    tuple_state += (np.expand_dims(state_[agent_id], axis=0),)
                elif isinstance(state_[agent_id], torch.Tensor):
                    tuple_state += (state_[agent_id].unsqueeze(0),)
                else:
                    import warnings
                    warnings.warn("State type not recognized")
            return tuple_state
        else:
            return np.expand_dims(state[agent_id], axis=0)

    @staticmethod
    def rearrange_states(states):
        states_ = tuple()
        for i in range(len(states[0])):
            states_ += (torch.cat([states[k][i] for k in range(len(states))]),)
        return states_

    def add(self, state, action, action_logprobs, reward, done, intrinsic_reward=None):
        for i in range(self.num_agents):
            int_reward = intrinsic_reward[i] if intrinsic_reward is not None else None
            self.memory[i].add(
                self.get_agent_state(state, agent_id=i),
                action[i],
                action_logprobs[i],
                reward[i],
                done[i],
                intrinsic_reward=int_reward
            )

    def __len__(self):
        # length = 0
        # for i in range(self.num_agents):
        #     length += len(self.memory[i])
        # return length
        return sum(len(mem) for mem in self.memory)

    def to_tensor(self):
        aggregated = defaultdict(list)
        for i in range(self.num_agents):
            agent_data = self.memory[i].to_tensor()  # dictionary
            for key, value in agent_data.items():
                aggregated[key].append(value)
        return dict(aggregated)

    def to_tensor2(self):
        states, actions, logprobs, rewards, not_dones = [], [], [], [], []
        for i in range(self.num_agents):
            state, action, logprob, reward, not_done = self.memory[i].to_tensor2()
            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            not_dones.append(not_done)
        return states, actions, logprobs, rewards, not_dones

    def change_horizon(self, new_horizon):
        for i in range(self.num_agents):
            self.memory[i].change_horizon(new_horizon)

    def clear_memory(self):
        for i in range(self.num_agents):
            self.memory[i].clear_memory()


class Memory(object):
    def __init__(self, action_dim=2, max_size=int(1e5)):
        self.max_size = max_size
        self.action_dim = action_dim
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(max_size)]
        self.action = np.zeros((max_size, action_dim))
        self.logprobs = np.zeros((max_size,))
        self.reward = np.zeros((max_size,))
        self.not_done = np.zeros((max_size,))

        # Optional
        self.intrinsic_reward = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self,
            state,
            action,
            action_logprobs,
            reward,
            done,
            intrinsic_reward=None):
        """
        Add a new experience to the memory.
        :param intrinsic_reward: Optional intrinsic reward (float)
        """
        self.state[self.ptr] = tuple(s for s in state)
        self.action[self.ptr] = action
        self.logprobs[self.ptr] = action_logprobs
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        if intrinsic_reward is not None:
            if self.intrinsic_reward is None:
                self.intrinsic_reward = np.zeros((self.max_size,))
            self.intrinsic_reward[self.ptr] = intrinsic_reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def to_tensor(self):
        """
        Return a dict with keys: 'state', 'action', 'logprobs', 'reward', 'not_done'
        And if intrinsic reward is available, 'intrinsic_reward'
        """
        data = {}

        state_tuple = tuple(torch.FloatTensor(np.array(state)).squeeze(1).to(self.device) for state in zip(*self.state[:self.size]))
        data['state'] = state_tuple
        data['action'] = torch.FloatTensor(self.action[:self.size]).to(self.device)
        data['logprobs'] = torch.FloatTensor(self.logprobs[:self.size]).to(self.device)
        data['reward'] = torch.FloatTensor(self.reward[:self.size]).to(self.device)
        data['not_done'] = torch.FloatTensor(self.not_done[:self.size]).to(self.device)

        # Only add intrinsic reward if it is available
        if self.intrinsic_reward is not None:
            data['intrinsic_reward'] = torch.FloatTensor(self.intrinsic_reward[:self.size]).to(self.device)

        return data

    def to_tensor2(self):
        return tuple(torch.FloatTensor(np.array(state)).squeeze(1).to(self.device) for state in zip(*self.state[:self.size])), \
               torch.FloatTensor(self.action[:self.size]).to(self.device), \
               torch.FloatTensor(self.logprobs[:self.size]).to(self.device), \
               torch.FloatTensor(self.reward[:self.size]).to(self.device), \
               torch.FloatTensor(self.not_done[:self.size]).to(self.device)

    def change_horizon(self, new_horizon):
        self.max_size = new_horizon
        self.state = [0 for _ in range(self.max_size)]
        self.action = np.zeros((self.max_size, self.action_dim))
        self.logprobs = np.zeros((self.max_size,))
        self.reward = np.zeros((self.max_size,))
        self.not_done = np.zeros((self.max_size,))

        # Reset optional fields
        self.intrinsic_reward = None

        self.clear_memory()

    def clear_memory(self):
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(self.max_size)]
        self.action.fill(0)
        self.logprobs.fill(0)
        self.reward.fill(0)
        self.not_done.fill(0)
        if self.intrinsic_reward is not None:
            self.intrinsic_reward.fill(0)
