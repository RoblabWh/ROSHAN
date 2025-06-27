import torch
import numpy as np
from collections import defaultdict

from transformers.models.deta.image_processing_deta import masks_to_boxes


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
    def rearrange_masks(masks):
        if not masks:
            return None
        # Concatenate masks along the batch dimension
        masks_ = tuple()
        for i in range(len(masks[0])):
            masks_ += (torch.cat([masks[k][i] for k in range(len(masks))]),)
        return masks_

    @staticmethod
    def rearrange_tensor(tensors):
        # Rearranges a list of simple tensors (like actions, rewards, etc.)
        return torch.cat(tensors, dim=0)

    @staticmethod
    def reconstruct_from_mask(padded_data, mask):
        """
        Given padded data [B, L, ...] and a mask [B, L],
        returns a list of tensors (one per batch), each [num_valid, ...].
        If mask is empty (non-padded data), just returns split tensors along the batch dim.
        """
        # Convert to torch if needed
        if isinstance(padded_data, np.ndarray):
            padded_data = torch.from_numpy(padded_data)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if mask.numel() == 0:
            # No mask: just return batch elements
            return [padded_data[i] for i in range(padded_data.shape[0])]
        else:
            # For each batch, select only valid entries
            out = []
            for i in range(padded_data.shape[0]):
                valid_idx = mask[i].bool()
                out.append(padded_data[i][valid_idx])
            return out

    def add(self, state, action, action_logprobs, reward, done, next_obs=None, intrinsic_reward=None):
        for i in range(self.num_agents):
            int_reward = intrinsic_reward[i] if intrinsic_reward is not None else None
            n_obs_ = self.get_agent_state(next_obs, agent_id=i) if next_obs is not None else None
            action_logprobs_ = action_logprobs[i] if action_logprobs is not None else None
            self.memory[i].add(
                self.get_agent_state(state, agent_id=i),
                action[i],
                action_logprobs_,
                reward[i],
                done[i],
                next_obs=n_obs_,
                intrinsic_reward=int_reward
            )

    def __len__(self):
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
        self.action = np.zeros((self.max_size, action_dim)) if isinstance(action_dim, int) else np.zeros((self.max_size, *action_dim))
        self.logprobs = np.zeros((self.max_size,))
        self.reward = np.zeros((self.max_size,))
        self.not_done = np.zeros((self.max_size,))
        self.masks = []

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

    def create_state_tuple(self, state_fields):
        """
        Create a tuple of states from the input state.
        This is used to ensure that the state is in the correct format.
        """
        state_tuple = []
        masks = []

        for i, field_list in enumerate(state_fields):
            # Convert all to numpy arrays, preserving singleton dims
            arr_list = [np.array(f) for f in field_list]
            # Find all shapes (should be e.g. (1,1,1,2) for drones, (1,1,NumFires,2) for fires)
            shapes = [a.shape for a in arr_list]
            # For each axis, get set of sizes
            axis_sizes = list(zip(*shapes))
            size_sets = [set(sizes) for sizes in axis_sizes]

            # Find axes with more than 1 unique size: these are variable-length axes
            # We'll pad on the first such axis we find (for most RL cases, only one axis is variable)
            variable_axes = [ax for ax, sizes in enumerate(size_sets) if len(sizes) > 1]
            if not variable_axes:
                # No variable axes: just stack normally, keep singleton dims
                t = torch.FloatTensor(np.stack(arr_list)).to(self.device)
                state_tuple.append(t)
                masks.append([])
            else:
                # There is at least one variable axis, pad along that axis
                axis = variable_axes[0]
                max_len = max(s[axis] for s in shapes)
                padded = []
                mask = []
                for arr in arr_list:
                    valid_len = arr.shape[axis]
                    # Build mask: 1 for valid, 0 for padded
                    mask_arr = np.zeros((max_len,), dtype=np.float32)
                    mask_arr[:valid_len] = 1.0
                    mask.append(mask_arr)
                    # Pad data as before
                    pad_width = [(0, 0)] * arr.ndim
                    pad_width[axis] = (0, max_len - valid_len)
                    arr = np.pad(arr, pad_width, mode='constant', constant_values=-1)
                    padded.append(arr)
                t = torch.FloatTensor(np.stack(padded)).to(self.device)
                mask_t = torch.FloatTensor(np.stack(mask)).to(self.device)  # [batch, max_len]
                masks.append(mask_t)
                state_tuple.append(t)

        return tuple(state_tuple), masks

    def to_tensor(self):
        """
        Return a dict with keys: 'state', 'action', 'logprobs', 'reward', 'not_done'
        And if intrinsic reward is available, 'intrinsic_reward'
        """
        data = {}

        # Transpose the state buffer: now tuples are grouped per state index
        state_fields = list(zip(*self.state[:self.size]))
        state_tuple, masks = self.create_state_tuple(state_fields)

        # state_tuple = tuple(torch.FloatTensor(np.array(state)).squeeze(1).to(self.device) for state in zip(*self.state[:self.size]))
        data['state'] = state_tuple
        data['masks'] = masks  # Store masks for variable-length states
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
        self.masks = []
        if self.intrinsic_reward is not None:
            self.intrinsic_reward.fill(0)
