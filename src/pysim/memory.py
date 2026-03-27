import torch
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple, Optional

def _to_cpu_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    # Keep numpy arrays
    return np.asarray(x) if not isinstance(x, np.ndarray) else x

class SwarmMemory(object):
    def __init__(self, num_agents=2, action_dim=2, max_size=int(1e5), use_intrinsic_reward=False, use_next_obs=False):
        self.num_agents = num_agents
        self.memory = [Memory(action_dim=action_dim, max_size=max_size, use_intrinsic_reward=use_intrinsic_reward, use_next_obs=use_next_obs) for _ in range(num_agents)]
        self._cached_cumsum = None  # Cached for sample_batch

    def save(self, path: str):
        pkg = {
            "version": 1,
            "num_agents": self.num_agents,
            "memories": [m.state_dict() for m in self.memory],
        }
        torch.save(pkg, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "SwarmMemory":
        pkg = torch.load(path, map_location=map_location)
        num_agents = int(pkg["num_agents"])
        # Recreate each Memory using its stored config
        mems = []
        for mstate in pkg["memories"]:
            m = Memory(
                action_dim=mstate["action_dim"],
                max_size=mstate["max_size"],
                use_intrinsic_reward=mstate["use_intrinsic_reward"],
                use_next_obs=mstate["use_next_obs"],
            )
            m.load_state_dict(mstate)
            mems.append(m)

        # Construct SwarmMemory and plug memories in
        sm = cls(num_agents=num_agents,
                 action_dim=mems[0].action_dim,
                 max_size=mems[0].max_size,
                 use_intrinsic_reward=mems[0].use_intrinsic_reward,
                 use_next_obs=mems[0].use_next_obs)
        sm.memory = mems
        return sm

    @staticmethod
    def get_agent_state(state, agent_id):
        if hasattr(state, 'keys') or isinstance(state, dict):
            # Dict-based path (ObservationDict or plain dict)
            return {
                k: (np.expand_dims(v[agent_id], axis=0) if isinstance(v, np.ndarray)
                    else v[agent_id].unsqueeze(0))
                for k, v in state.items()
            }
        elif isinstance(state, tuple):
            return tuple(
                np.expand_dims(s[agent_id], axis=0) if isinstance(s[agent_id], np.ndarray)
                else s[agent_id].unsqueeze(0)
                for s in state
            )
        else:
            return np.expand_dims(state[agent_id], axis=0)

    @staticmethod
    def rearrange_states(states):
        if states and (hasattr(states[0], 'keys') or isinstance(states[0], dict)):
            # Dict-based path: merge dicts along batch dim
            keys = states[0].keys()
            return {k: torch.cat([s[k] for s in states], dim=0) for k in keys}
        # Legacy tuple path: transpose list-of-tuples -> tuple-of-lists
        fields = list(zip(*states))  # length K, each is a list of length T
        return tuple(torch.cat(field_tensors, dim=0) for field_tensors in fields)

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
        self._cached_cumsum = None  # Invalidate sample_batch cache
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

    def _get_batch_idx(self, batch_size):
        N = self.__len__()
        assert N >= batch_size, f"Not enough samples: have {N}, need {batch_size}"
        idxs = np.random.randint(0, N, size=(batch_size,), dtype=np.int64)
        return idxs

    def sample_batch(self, batch_size: int):
        """
        Uniform over all transitions across all agents.
        Returns a dict like your to_tensor(), but only for the sampled rows.
        """
        # Build cumulative lengths to map global -> (agent, local)
        # Cache cumsum/starts to avoid recomputing when memory size hasn't changed
        if self._cached_cumsum is None:
            lengths = np.array([len(m) for m in self.memory], dtype=np.int64)
            self._cached_cumsum = np.cumsum(lengths)
        cumsum = self._cached_cumsum
        # sample global indices
        gidx = self._get_batch_idx(batch_size)
        # find which agent each global index belongs to
        agent_ids = np.searchsorted(cumsum, gidx, side='right')
        # local index within that agent
        starts = np.concatenate(([0], cumsum[:-1]))
        local_idx = gidx - starts[agent_ids]

        # bucket local indices per agent to minimize per-agent overhead
        idx_per_agent = {a: [] for a in range(self.num_agents)}
        for a, li in zip(agent_ids, local_idx):
            idx_per_agent[int(a)].append(int(li))

        # Gather per agent, then concat
        collected = []
        for a, idxs in idx_per_agent.items():
            if not idxs:
                continue
            batch_a = self.memory[a].gather(idxs)
            collected.append((a, batch_a))

        # Now merge all per-agent dicts
        # We must handle tuples (state, next_obs) by field-wise concat.
        def concat_field(values, is_tuple=False):
            if is_tuple:
                # values: list of tuples of tensors; need field-wise cat
                num_fields = len(values[0])
                return tuple(torch.cat([v[i] for v in values], dim=0) for i in range(num_fields))
            else:
                return torch.cat(values, dim=0)

        # Collect keys
        keys = set().union(*[d.keys() for _, d in collected])

        merged = {}
        for key in keys:
            vals = [d[key] for _, d in collected if key in d and d[key] is not None]
            if not vals:
                merged[key] = None
                continue
            if isinstance(vals[0], dict):
                dict_keys = vals[0].keys()
                merged[key] = {k: torch.cat([v[k] for v in vals], dim=0) for k in dict_keys}
            elif isinstance(vals[0], tuple):
                merged[key] = concat_field(vals, is_tuple=True)
            else:
                merged[key] = concat_field(vals, is_tuple=False)

        # # Optional: shuffle within the batch to remove agent clustering artifacts
        # B = merged["action"].shape[0]
        # perm = torch.randperm(B, device=self.device)
        # # apply same perm to all tensor fields
        # def apply_perm(x):
        #     if x is None: return None
        #     if isinstance(x, tuple):
        #         return tuple(xx[perm] for xx in x)
        #     return x[perm]
        #
        # for k in merged:
        #     merged[k] = apply_perm(merged[k])

        return merged

    def change_horizon(self, new_horizon):
        for i in range(self.num_agents):
            self.memory[i].change_horizon(new_horizon)

    def clear_memory(self):
        self._cached_cumsum = None
        for i in range(self.num_agents):
            self.memory[i].clear_memory()


class Memory(object):
    __slots__ = ('max_size', 'action_dim', 'ptr', 'batch_ptr', 'size', '_pinned_refs',
                 '_tensor_cache', '_tensor_cache_valid',
                 'state', 'action', 'logprobs', 'reward', 'not_done', 'masks',
                 'use_intrinsic_reward', 'intrinsic_reward',
                 'use_next_obs', 'next_obs', 'device')

    def _alloc_buffer(self, shape):
        """Allocate numpy buffer, pinned if CUDA available."""
        if torch.cuda.is_available():
            t = torch.zeros(shape, dtype=torch.float32, pin_memory=True)
            self._pinned_refs.append(t)  # prevent GC
            return t.numpy()
        return np.zeros(shape)

    def __init__(self, action_dim=2, max_size=int(1e5), use_intrinsic_reward=False, use_next_obs=False):
        self.max_size = int(max_size)
        self.action_dim = action_dim
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0
        self._pinned_refs = []
        self._tensor_cache = None  # Cached to_tensor() result
        self._tensor_cache_valid = False  # Dirty flag

        self.state = [0 for _ in range(self.max_size)]
        self.action = self._alloc_buffer((self.max_size, action_dim)) if isinstance(action_dim, int) else self._alloc_buffer((self.max_size, *action_dim))
        self.logprobs = self._alloc_buffer((self.max_size,)) if isinstance(action_dim, int) else self._alloc_buffer((self.max_size, action_dim[0]))
        self.reward = self._alloc_buffer((self.max_size,))
        self.not_done = self._alloc_buffer((self.max_size,))
        self.masks = []

        # Optional
        self.use_intrinsic_reward = use_intrinsic_reward
        if self.use_intrinsic_reward:
            self.intrinsic_reward = self._alloc_buffer((self.max_size,))
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


    def state_dict(self) -> Dict[str, Any]:
        """Serialize *only* the filled part of the buffer ([:self.size])."""
        # Convert nested tuples/dicts of states/next_obs to CPU tensors/ndarrays
        def _convert_state_list(lst: Optional[List]):
            if lst is None:
                return None
            result = []
            for sample in lst[:self.size]:
                if isinstance(sample, dict):
                    result.append({k: _to_cpu_numpy(v) for k, v in sample.items()})
                elif isinstance(sample, tuple):
                    result.append(tuple(_to_cpu_numpy(field) for field in sample))
                else:
                    result.append(sample)
            return result

        data = {
            "version": 1,
            "max_size": self.max_size,
            "action_dim": self.action_dim,
            "ptr": self.ptr,
            "size": self.size,
            "use_intrinsic_reward": self.use_intrinsic_reward,
            "use_next_obs": self.use_next_obs,

            # Buffers truncated to current size
            "state": _convert_state_list(self.state),
            "action": self.action[:self.size],
            "logprobs": self.logprobs[:self.size],
            "reward": self.reward[:self.size],
            "not_done": self.not_done[:self.size],
            # masks is unused in your current flow; keep for completeness
            "masks": self.masks,

            "intrinsic_reward": (
                None if not self.use_intrinsic_reward else self.intrinsic_reward[:self.size]
            ),
            "next_obs": _convert_state_list(self.next_obs) if self.use_next_obs else None,
        }
        return data

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore buffers and pointers. Resizes storage if needed."""
        # Reallocate to max_size from file (keeps spare capacity)
        self.max_size = int(state["max_size"])
        self.action_dim = state["action_dim"]
        self.use_intrinsic_reward = bool(state["use_intrinsic_reward"])
        self.use_next_obs = bool(state["use_next_obs"])

        # Re-init backing storage
        self._pinned_refs = []
        self.state = [0 for _ in range(self.max_size)]
        self.action = self._alloc_buffer((self.max_size, self.action.shape[1])) if isinstance(self.action_dim, int) else self._alloc_buffer((self.max_size, *self.action_dim))
        self.logprobs = self._alloc_buffer((self.max_size,)) if isinstance(self.action_dim, int) else self._alloc_buffer((self.max_size, self.action_dim[0]))
        self.reward = self._alloc_buffer((self.max_size,))
        self.not_done = self._alloc_buffer((self.max_size,))
        self.masks = state.get("masks", [])

        if self.use_intrinsic_reward:
            self.intrinsic_reward = self._alloc_buffer((self.max_size,))
        else:
            self.intrinsic_reward = None

        if self.use_next_obs:
            self.next_obs = [0 for _ in range(self.max_size)]
        else:
            self.next_obs = None

        # Fill with the saved content
        n = int(state["size"])
        self.size = n
        self.ptr = int(state["ptr"])

        # States / next_obs are lists length n
        for i in range(n):
            s = state["state"][i]
            if isinstance(s, dict):
                self.state[i] = {k: _to_cpu_numpy(v) for k, v in s.items()}
            else:
                self.state[i] = tuple(_to_cpu_numpy(f) for f in s)

        self.action[:n] = state["action"]
        self.logprobs[:n] = state["logprobs"]
        self.reward[:n] = state["reward"]
        self.not_done[:n] = state["not_done"]

        if self.use_intrinsic_reward and state["intrinsic_reward"] is not None:
            self.intrinsic_reward[:n] = state["intrinsic_reward"]

        if self.use_next_obs and state["next_obs"] is not None:
            for i in range(n):
                s = state["next_obs"][i]
                if isinstance(s, dict):
                    self.next_obs[i] = {k: _to_cpu_numpy(v) for k, v in s.items()}
                else:
                    self.next_obs[i] = tuple(_to_cpu_numpy(f) for f in s)

    def save(self, path: str):
        """Torch’s pickler handles tensors + numpy cleanly."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "Memory":
        state = torch.load(path, map_location=map_location)
        # Build with saved config to allocate correctly
        mem = cls(
            action_dim=state["action_dim"],
            max_size=state["max_size"],
            use_intrinsic_reward=state["use_intrinsic_reward"],
            use_next_obs=state["use_next_obs"],
        )
        mem.load_state_dict(state)
        return mem

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
        if hasattr(state, 'keys') or isinstance(state, dict):
            self.state[self.ptr] = dict(state)
        else:
            self.state[self.ptr] = tuple(s for s in state)
        self.action[self.ptr] = action
        self.logprobs[self.ptr] = action_logprobs
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        if self.use_intrinsic_reward:
            self.intrinsic_reward[self.ptr] = intrinsic_reward

        if self.use_next_obs:
            if hasattr(next_obs, 'keys') or isinstance(next_obs, dict):
                self.next_obs[self.ptr] = dict(next_obs)
            else:
                self.next_obs[self.ptr] = tuple(s for s in next_obs)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self._tensor_cache_valid = False  # Invalidate cache on new data

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
            if len(shapes[0]) > 1 and all(s[:1] == (1,) for s in shapes):
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

    def _is_dict_state(self):
        """Check if stored states are dict-based (new schema pipeline)."""
        for i in range(self.size):
            s = self.state[i]
            if s != 0:
                return isinstance(s, dict)
        return False

    def _to_tensor_dict_states(self):
        """Convert dict-based states to a dict of stacked tensors."""
        states = self.state[:self.size]
        keys = states[0].keys()
        result = {}
        mask = torch.empty(0, dtype=torch.bool, device=self.device)
        for k in keys:
            field_list = [s[k] for s in states]
            tensor_list = [torch.as_tensor(f) if getattr(f, 'dtype', None) == bool
                          else torch.as_tensor(f, dtype=torch.float32) for f in field_list]
            shapes = [t.shape for t in tensor_list]

            # Strip leading dim=1 from batch storage
            if len(shapes[0]) > 1 and all(s[:1] == (1,) for s in shapes):
                tensor_list = [t.squeeze(0) for t in tensor_list]
                shapes = [t.shape for t in tensor_list]

            # Check for variable-length axis
            variable_axis = None
            for ax in range(len(shapes[0])):
                if len({s[ax] for s in shapes}) > 1:
                    variable_axis = ax
                    break

            if variable_axis is None:
                result[k] = torch.stack(tensor_list).to(self.device)
            else:
                axis = variable_axis
                lengths = torch.tensor([t.shape[axis] for t in tensor_list], device=self.device)
                permute_order = [axis] + [i for i in range(len(shapes[0])) if i != axis]
                dims_rest = permute_order[1:]
                rearranged = [t.permute(permute_order) for t in tensor_list]
                padded = pad_sequence(rearranged, batch_first=True)
                max_len = padded.size(1)
                mask = (
                    torch.arange(max_len, device=self.device).expand(len(lengths), max_len)
                    >= lengths.unsqueeze(1)
                )
                order = [0]
                for j in range(len(shapes[0])):
                    if j == axis:
                        order.append(1)
                    else:
                        ki = dims_rest.index(j)
                        order.append(ki + 2)
                padded = padded.permute(order)
                result[k] = padded.to(self.device)

        return result, mask

    def to_tensor(self):
        """
        Return a dict with keys: 'state', 'action', 'logprobs', 'reward', 'not_done'
        And if intrinsic reward is available, 'intrinsic_reward'
        Uses dirty-flag caching: only rebuilds when data has changed since last call.
        """
        if self._tensor_cache_valid and self._tensor_cache is not None:
            return self._tensor_cache

        data = {}

        if self._is_dict_state():
            state_dict, mask = self._to_tensor_dict_states()
            data['state'] = state_dict
            data['mask'] = mask
        else:
            # Legacy tuple path: transpose the state buffer
            state_fields = list(zip(*self.state[:self.size]))
            state_tuple, mask = self.create_state_tuple(state_fields)
            data['state'] = state_tuple
            data['mask'] = mask

        data['action'] = torch.as_tensor(self.action[:self.size], dtype=torch.float32).to(self.device, non_blocking=True)
        data['logprobs'] = torch.as_tensor(self.logprobs[:self.size], dtype=torch.float32).to(self.device, non_blocking=True)
        data['reward'] = torch.as_tensor(self.reward[:self.size], dtype=torch.float32).to(self.device, non_blocking=True)
        data['not_done'] = torch.as_tensor(self.not_done[:self.size], dtype=torch.float32).to(self.device, non_blocking=True)

        # Only add intrinsic reward if it is available
        if self.use_intrinsic_reward:
            data['intrinsic_reward'] = torch.as_tensor(self.intrinsic_reward[:self.size], dtype=torch.float32).to(self.device, non_blocking=True)

        # Only add next_obs if it is available
        if self.use_next_obs:
            if self._is_dict_state():
                next_states = self.next_obs[:self.size]
                if next_states and isinstance(next_states[0], dict):
                    keys = next_states[0].keys()
                    data['next_obs'] = {
                        k: torch.as_tensor(np.array([s[k] for s in next_states]), dtype=torch.float32).squeeze(1).to(self.device, non_blocking=True)
                        for k in keys
                    }
                else:
                    next_obs_tuple = tuple(torch.as_tensor(np.array(state), dtype=torch.float32).squeeze(1).to(self.device, non_blocking=True) for state in zip(*next_states))
                    data['next_obs'] = next_obs_tuple
            else:
                next_obs_tuple = tuple(torch.as_tensor(np.array(state), dtype=torch.float32).squeeze(1).to(self.device, non_blocking=True) for state in zip(*self.next_obs[:self.size]))
                data['next_obs'] = next_obs_tuple

        self._tensor_cache = data
        self._tensor_cache_valid = True
        return data

    # return ONLY the selected indices as tensors on device
    def gather(self, idx, *, as_tensors=True):
        # idx: 1D LongTensor or numpy array of indices into this buffer
        selected_states = [self.state[i] for i in idx]

        if selected_states and isinstance(selected_states[0], dict):
            # Dict-based path
            keys = selected_states[0].keys()
            state_dict = {}
            mask = torch.empty(0, dtype=torch.bool, device=self.device)
            for k in keys:
                field_list = [s[k] for s in selected_states]
                tensor_list = [torch.as_tensor(f) if getattr(f, 'dtype', None) == bool
                              else torch.as_tensor(f, dtype=torch.float32) for f in field_list]
                shapes = [t.shape for t in tensor_list]
                if len(shapes[0]) > 1 and all(s[:1] == (1,) for s in shapes):
                    tensor_list = [t.squeeze(0) for t in tensor_list]
                state_dict[k] = torch.stack(tensor_list).to(self.device)
            state_result = state_dict
        else:
            # Legacy tuple path
            state_fields = list(zip(*selected_states))
            state_result, mask = self.create_state_tuple(state_fields)

        idx_arr = np.asarray(idx)
        actions   = torch.as_tensor(self.action[idx_arr], dtype=torch.float32, device=self.device)
        logprobs  = torch.as_tensor(self.logprobs[idx_arr], dtype=torch.float32, device=self.device)
        rewards   = torch.as_tensor(self.reward[idx_arr], dtype=torch.float32, device=self.device)
        not_dones = torch.as_tensor(self.not_done[idx_arr], dtype=torch.float32, device=self.device)

        out = {
            "state": state_result,
            "mask": mask if not isinstance(state_result, dict) else torch.empty(0, dtype=torch.bool, device=self.device),
            "action": actions,
            "logprobs": logprobs,
            "reward": rewards,
            "not_done": not_dones,
        }

        if self.use_intrinsic_reward and len(self.intrinsic_reward) > 0:
            out["intrinsic_reward"] = torch.as_tensor(
                self.intrinsic_reward[idx_arr], dtype=torch.float32, device=self.device
            )

        if self.use_next_obs and len(self.next_obs) > 0:
            selected_next = [self.next_obs[i] for i in idx]
            if selected_next and isinstance(selected_next[0], dict):
                keys = selected_next[0].keys()
                out["next_obs"] = {
                    k: torch.stack([torch.as_tensor(s[k], dtype=torch.float32) for s in selected_next]).to(self.device)
                    for k in keys
                }
            else:
                next_fields = list(zip(*selected_next))
                next_state_tuple, next_mask = self.create_state_tuple(next_fields)
                out["next_obs"] = next_state_tuple

        return out

    def _sample_batch(self, idxs):

        # Build state tuple correctly
        state_fields = list(zip(*[self.state[i] for i in idxs]))
        state_batch = tuple(torch.as_tensor(np.array(field), dtype=torch.float32).squeeze(1).to(self.device, non_blocking=True) for field in state_fields)

        action_batch = torch.as_tensor(self.action[idxs], dtype=torch.float32).to(self.device, non_blocking=True)
        reward_batch = torch.as_tensor(self.reward[idxs], dtype=torch.float32).to(self.device, non_blocking=True)
        not_done_batch = torch.as_tensor(self.not_done[idxs], dtype=torch.float32).to(self.device, non_blocking=True)

        batch = {
            'state': state_batch,
            'action': action_batch,
            'reward': reward_batch,
            'not_done': not_done_batch
        }

        if self.use_next_obs:
            next_obs_fields = list(zip(*[self.next_obs[i] for i in idxs]))
            next_obs_batch = tuple(torch.as_tensor(np.array(field), dtype=torch.float32).squeeze(1).to(self.device, non_blocking=True) for field in next_obs_fields)
            batch['next_state'] = next_obs_batch

        return batch

    def change_horizon(self, new_horizon):
        self.max_size = new_horizon
        self._pinned_refs = []
        self.state = [0 for _ in range(self.max_size)]
        self.action = self._alloc_buffer((self.max_size, self.action_dim))
        self.logprobs = self._alloc_buffer((self.max_size,))
        self.reward = self._alloc_buffer((self.max_size,))
        self.not_done = self._alloc_buffer((self.max_size,))

        # Reset optional fields
        self.intrinsic_reward = None
        self._tensor_cache = None
        self._tensor_cache_valid = False

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
        self._tensor_cache = None
        self._tensor_cache_valid = False
