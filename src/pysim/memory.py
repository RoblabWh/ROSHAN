import torch
import numpy as np
from collections import defaultdict


class SwarmMemory(object):
    def __init__(self, num_agents=2, action_dim=2, max_size=int(1e5), use_intrinsic_reward=False, use_next_obs=False):
        self.num_agents = num_agents
        self.memory = [Memory(action_dim=action_dim, max_size=max_size, use_intrinsic_reward=use_intrinsic_reward, use_next_obs=use_next_obs) for _ in range(num_agents)]

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

    @staticmethod
    def rearrange_tensor(tensors):
        # Rearranges a list of simple tensors (like actions, rewards, etc.)
        return torch.cat(tensors, dim=0)

    def add(self, state, action, action_logprobs, reward, done, next_obs=None, intrinsic_reward=None):
        for i in range(self.num_agents):
            int_reward = intrinsic_reward[i] if intrinsic_reward is not None else None
            n_obs_ = self.get_agent_state(next_obs, agent_id=i) if next_obs is not None else None
            self.memory[i].add(
                self.get_agent_state(state, agent_id=i),
                action[i],
                action_logprobs[i],
                reward[i],
                done[i],
                next_obs=n_obs_,
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

    def sample_batch(self, batch_size):
        batch = defaultdict(list)
        for i in range(self.num_agents):
            agent_batch = self.memory[i].sample_batch(batch_size)
            for key, value in agent_batch.items():
                batch[key].append(value)
        return dict(batch)

    def change_horizon(self, new_horizon):
        for i in range(self.num_agents):
            self.memory[i].change_horizon(new_horizon)

    def clear_memory(self):
        for i in range(self.num_agents):
            self.memory[i].clear_memory()


class Memory(object):
    def __init__(self, action_dim=2, max_size=int(1e5), use_intrinsic_reward=False, use_next_obs=False):
        self.max_size = int(max_size)
        self.action_dim = action_dim
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(self.max_size)]
        self.action = np.zeros((self.max_size, action_dim))
        self.logprobs = np.zeros((self.max_size,))
        self.reward = np.zeros((self.max_size,))
        self.not_done = np.zeros((self.max_size,))

        # Optional
        self.use_intrinsic_reward = use_intrinsic_reward
        if self.use_intrinsic_reward:
            self.intrinsic_reward = np.zeros((self.max_size,))
        else:
            self.intrinsic_reward = None

        self.use_next_obs = use_next_obs
        if self.use_next_obs:
            self.next_obs = [0 for _ in range(self.max_size)]
        else:
            self.next_obs = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self,
            state,
            action,
            action_logprobs,
            reward,
            done,
            next_obs=None,
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

        if self.use_intrinsic_reward:
            self.intrinsic_reward[self.ptr] = intrinsic_reward

        if self.use_next_obs:
            self.next_obs[self.ptr] = tuple(s for s in next_obs)

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
        if self.use_intrinsic_reward:
            data['intrinsic_reward'] = torch.FloatTensor(self.intrinsic_reward[:self.size]).to(self.device)

        # Only add next_obs if it is available
        if self.use_next_obs:
            next_obs_tuple = tuple(torch.FloatTensor(np.array(state)).squeeze(1).to(self.device) for state in zip(*self.next_obs[:self.size]))
            data['next_obs'] = next_obs_tuple

        return data

    # def sample_batch(self, batch_size):
    #     idxs = np.random.randint(0, self.size, size=batch_size)
    #
    #     state_batch = tuple(torch.FloatTensor(np.array([self.state[i] for i in idxs])).squeeze(1).to(self.device))
    #     action_batch = torch.FloatTensor(self.action[idxs]).to(self.device)
    #     reward_batch = torch.FloatTensor(self.reward[idxs]).to(self.device)
    #     not_done_batch = torch.FloatTensor(self.not_done[idxs]).to(self.device)
    #
    #     batch = {
    #         'state': state_batch,
    #         'action': action_batch,
    #         'reward': reward_batch,
    #         'not_done': not_done_batch
    #     }
    #
    #     if self.use_next_obs:
    #         next_obs_batch = tuple(
    #             torch.FloatTensor(np.array([self.next_obs[i] for i in idxs])).squeeze(1).to(self.device))
    #         batch['next_state'] = next_obs_batch
    #
    #     return batch
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # Build state tuple correctly
        state_fields = list(zip(*[self.state[i] for i in idxs]))
        state_batch = tuple(torch.FloatTensor(np.array(field)).squeeze(1).to(self.device) for field in state_fields)

        action_batch = torch.FloatTensor(self.action[idxs]).to(self.device)
        reward_batch = torch.FloatTensor(self.reward[idxs]).to(self.device)
        not_done_batch = torch.FloatTensor(self.not_done[idxs]).to(self.device)

        batch = {
            'state': state_batch,
            'action': action_batch,
            'reward': reward_batch,
            'not_done': not_done_batch
        }

        if self.use_next_obs:
            next_obs_fields = list(zip(*[self.next_obs[i] for i in idxs]))
            next_obs_batch = tuple(
                torch.FloatTensor(np.array(field)).squeeze(1).to(self.device) for field in next_obs_fields)
            batch['next_state'] = next_obs_batch

        return batch

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
