import torch
import numpy as np


class Memory(object):
    def __init__(self, action_dim=3, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(max_size)]
        self.action = np.zeros((max_size, action_dim))
        self.logprobs = np.zeros((max_size,))
        self.reward = np.zeros((max_size,))
        self.not_done = np.zeros((max_size,))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self, state, action, action_logprobs, reward, done):
        for i in range(len(done)):
            self.state[self.ptr] = tuple(s[i] for s in state)
            self.action[self.ptr] = action[i]
            self.logprobs[self.ptr] = action_logprobs[i]
            self.reward[self.ptr] = reward[i]
            self.not_done[self.ptr] = 1. - float(done[i])

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def to_tensor(self):
        return tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in zip(*self.state[:self.size])), \
               torch.FloatTensor(self.action[:self.size]).to(self.device), \
               torch.FloatTensor(self.logprobs[:self.size]).to(self.device), \
               torch.FloatTensor(self.reward[:self.size]).to(self.device), \
               torch.FloatTensor(self.not_done[:self.size]).to(self.device)

    def clear_memory(self):
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(self.max_size)]
        self.action.fill(0)
        self.logprobs.fill(0)
        self.reward.fill(0)
        self.not_done.fill(0)
