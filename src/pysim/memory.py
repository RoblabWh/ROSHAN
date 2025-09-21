import torch
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

# Optional dependency used elsewhere in the project.
try:
    from transformers.models.deta.image_processing_deta import masks_to_boxes
except Exception:  # pragma: no cover - runtime availability only
    masks_to_boxes = None


class SwarmMemory(object):
    def __init__(self, num_agents=2, action_dim=2, max_size=int(1e5), use_intrinsic_reward=False, use_next_obs=False):
        self.num_agents = num_agents
        self.memory = [Memory(action_dim=action_dim, max_size=max_size, use_intrinsic_reward=use_intrinsic_reward, use_next_obs=use_next_obs) for _ in range(num_agents)]

    @staticmethod
    # TODO WATCH THIS!! DOES THIS EVEN WORK AS INTENDED??? I THINK IT DOES??
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
        """Pad variable-length fields into batched tensors and a mask.

        Parameters
        ----------
        state_fields : Iterable[Iterable[Tensor]]
            Each element collects all samples of one field. Tensors within a
            field share the same rank and may vary in at most one axis length.
            Only the first axis with differing sizes is padded; fields with
            multiple varying axes are not currently supported.

        Returns
        -------
        tuple of torch.Tensor
            For each input field, a tensor of shape ``(B, *S)`` is returned
            where ``B`` equals the number of samples in the field and ``S`` is
            the original tensor shape. The axis that differs in length across
            samples is padded to ``L_max`` (the maximum length) and remains in
            its original position following the batch dimension. All tensors are
            placed on ``self.device``.
        torch.BoolTensor
            Mask with shape ``(B, L_max)`` on ``self.device`` where ``True``
            denotes padding and ``False`` denotes valid data. The mask's second
            dimension corresponds to the padded axis in the returned tensors.
        """
        state_tuple = []
        mask_tensor = torch.empty(0, dtype=torch.bool, device=self.device)

        for field_list in state_fields:
            tensor_list = [torch.as_tensor(f) if f.dtype==bool else torch.as_tensor(f, dtype=torch.float32) for f in field_list]#[torch.as_tensor(f) for f in field_list]
            shapes = [t.shape for t in tensor_list]

            # Strip extra dimensions from big brain storage solution
            if len(shapes[0]) > 0 and all(s[:1] == (1,) for s in shapes):
                tensor_list = [t.squeeze(0) for t in tensor_list]
                shapes = [t.shape for t in tensor_list]

            # Determine which axis varies in length across samples. We only pad
            # the *first* such axis encountered; remaining axes are assumed to
            # have consistent sizes. Supporting multiple variable axes would
            # require nested padding or a more general batching strategy.
            variable_axis = None
            for ax in range(len(shapes[0])):
                if len({s[ax] for s in shapes}) > 1:
                    variable_axis = ax
                    break  # stop at first variable axis

            if variable_axis is None:
                t = torch.stack(tensor_list).to(self.device)
                state_tuple.append(t)
            else:
                axis = variable_axis
                lengths = torch.tensor([t.shape[axis] for t in tensor_list], device=self.device)

                # pad_sequence operates on the leading dimension, so move the
                # variable axis to the front before padding and remember how to
                # restore the original layout afterwards
                permute_order = [axis] + [i for i in range(len(shapes[0])) if i != axis]
                dims_rest = permute_order[1:]
                rearranged = [t.permute(permute_order) for t in tensor_list]

                padded = pad_sequence(rearranged, batch_first=True)
                max_len = padded.size(1)

                mask_tensor = (
                    torch.arange(max_len, device=self.device).expand(len(lengths), max_len)
                    >= lengths.unsqueeze(1)
                )

                # Restore original axis order after padding. `order` first keeps
                # the batch dimension, then places the padded axis back in its
                # original position followed by the remaining dimensions.
                order = [0]
                for j in range(len(shapes[0])):
                    if j == axis:
                        order.append(1)
                    else:
                        k = dims_rest.index(j)
                        order.append(k + 2)
                padded = padded.permute(order)

                state_tuple.append(padded.to(self.device))

        return tuple(state_tuple), mask_tensor

    def to_tensor(self):
        """
        Return a dict with keys: 'state', 'action', 'logprobs', 'reward', 'not_done'
        And if intrinsic reward is available, 'intrinsic_reward'
        """
        data = {}

        # Transpose the state buffer: now tuples are grouped per state index
        state_fields = list(zip(*self.state[:self.size]))
        state_tuple, mask = self.create_state_tuple(state_fields)

        # state_tuple = tuple(torch.FloatTensor(np.array(state)).squeeze(1).to(self.device) for state in zip(*self.state[:self.size]))
        data['state'] = state_tuple
        data['mask'] = mask  # Store mask for variable-length states
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
