import torch
import numpy as np


# class Memory:
#     def __init__(self, max_size=int(1e6)):
#         self.capacity = 0
#         self.max_size = max_size
#         self.data = {
#             'states': [],
#             'actions': [],
#             'logprobs': [],
#             'next_states': [],
#             'rewards': [],
#             'is_terminals': []
#         }
#
#     def add(self, states, actions, logprobs, rewards, next_states, terminals):
#         items = {
#             'states': states,
#             'actions': actions,
#             'logprobs': logprobs,
#             'next_states': next_states,
#             'rewards': rewards,
#             'is_terminals': terminals
#         }
#
#         # Add data to the list
#         # TODO s[0] and v[0] are dirty solutions because ONE agent is used
#         for k, v in items.items():
#             if k in ['states', 'next_states']:
#                 self.data[k].append(tuple(s[0] for s in v))
#             elif k in ['is_terminals']:
#                 self.data[k].append(float(v[0]))
#             else:
#                 self.data[k].append(v[0])
#
#         self.capacity += len(actions)
#
#     def sample(self, batch_size):
#         indices = np.random.choice(self.capacity, batch_size, replace=False)
#         sampled_data = {k: [v[i] for i in indices] for k, v in self.data.items()}
#
#         tensors = {}
#         for k, v in sampled_data.items():
#             if k in ['states', 'next_states']:
#                 # Convert each state in the tuple to a torch tensor
#                 tensors[k] = tuple(torch.tensor(np.array(state), dtype=torch.float32) for state in zip(*v))
#             elif k in ['actions']:
#                 tensors[k] = torch.stack(v)
#             else:
#                 tensors[k] = torch.tensor(v)
#         return tensors['states'], tensors['actions'], tensors['rewards'], tensors['next_states'], tensors['is_terminals']
#
#     def clear_memory(self):
#         for k in self.data:
#             self.data[k] = []
#         self.capacity = 0
#
#     def to_numpy(self):
#         # Convert lists to numpy arrays
#         for k, v in self.data.items():
#             if k in ['states', 'next_states']:
#                 # Convert each state in the tuple to a numpy array
#                 self.data[k] = tuple(np.array(state) for state in zip(*v))
#             else:
#                 self.data[k] = np.array(v)
#
#     def to_tensors(self):
#         # Ensure data is in numpy format first
#         tensors = {}
#         for k, v in self.data.items():
#             if k in ['states', 'next_states']:
#                 # Convert each state in the tuple to a torch tensor
#                 tensors[k] = tuple(torch.tensor(np.array(state), dtype=torch.float32) for state in zip(*v))
#             else:
#                 tensors[k] = torch.tensor(v, dtype=torch.float32)
#         return tensors

class Memory(object):
    def __init__(self, action_dim=3, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(max_size)]
        self.action = np.zeros((max_size, action_dim))
        self.logprobs = np.zeros((max_size,))
        self.next_state = [0 for _ in range(max_size)]
        self.reward = np.zeros((max_size,))
        self.not_done = np.zeros((max_size,))
        self.masks = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self, state, action, action_logprobs, reward, next_state, done):
        for i in range(len(done)):
            self.state[self.ptr] = tuple(s[i] for s in state)
            self.action[self.ptr] = action[i]
            self.logprobs[self.ptr] = action_logprobs[i]
            self.next_state[self.ptr] = tuple(s[i] for s in next_state)
            self.reward[self.ptr] = reward[i]
            self.not_done[self.ptr] = 1. - float(done[i])

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def build_masks(self):
        for i in range(self.size):
            self.masks.append(self.not_done[i])

    def has_batches(self):
        return self.batch_ptr < (self.size - 1) # TODO little bit dirty I throw away one observation

    def to_tensor(self):
        return tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in zip(*self.state[:self.size])), \
               torch.FloatTensor(self.action[:self.size]).to(self.device), \
               torch.FloatTensor(self.logprobs[:self.size]).to(self.device), \
               torch.FloatTensor(self.reward[:self.size]).to(self.device), \
               tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in zip(*self.next_state[:self.size])), \
               torch.FloatTensor(self.masks[:self.size]).to(self.device)

    def next_batch(self, batch_size):
        batch_start = self.batch_ptr
        if self.batch_ptr + batch_size < self.size:
            batch_end = self.batch_ptr + batch_size
        else:
            batch_end = self.size
        self.batch_ptr = self.batch_ptr + batch_size
        return (
            tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in zip(*self.state[batch_start:batch_end])),
            torch.FloatTensor(self.action[batch_start:batch_end]).to(self.device),
            torch.FloatTensor(self.logprobs[batch_start:batch_end]).to(self.device),
            torch.FloatTensor(self.reward[batch_start:batch_end]).to(self.device),
            torch.FloatTensor(self.masks[batch_start:batch_end]).to(self.device)
        )

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        selected_states = [self.state[i] for i in ind]
        selected_next_states = [self.next_state[i] for i in ind]

        return (
            tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in zip(*selected_states)),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            tuple(torch.FloatTensor(np.array(next_state)).to(self.device) for next_state in zip(*selected_next_states)),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def clear_memory(self):
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(self.max_size)]
        self.action = np.zeros((self.max_size, 3))
        self.logprobs = np.zeros((self.max_size,))
        self.next_state = [0 for _ in range(self.max_size)]
        self.reward = np.zeros((self.max_size,))
        self.not_done = np.zeros((self.max_size,))
        self.masks = []
