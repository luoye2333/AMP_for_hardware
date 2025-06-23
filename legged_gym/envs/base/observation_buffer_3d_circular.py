# created by Xu Xin, 2025
import isaacgym
import torch
import numpy as np
from observation_buffer_3d import ObservationBuffer3D
from observation_buffer import ObservationBuffer

class ObservationBuffer3D_circular:
    """using a 3d tensor for storage (num_envs, num_obs, history_steps) \\
    0 = old obs -> end = new obs
    """
    def __init__(self, num_envs, num_obs, history_length, device):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.history_length = history_length
        self.device = device

        self.obs_history = torch.zeros((self.num_envs, self.num_obs, self.history_length), 
                                   device=self.device, dtype=torch.float)
        self.current_index = 0  # a common pointer for all envs

    def reset(self, reset_ids, new_obs):
        self.obs_history[reset_ids, :, :] = new_obs.unsqueeze(-1).repeat(1, 1, self.history_length)

    def insert(self, new_obs):
        # write the data to the pointer
        self.obs_history[:, :, self.current_index] = new_obs
        # update the pointer to a unwrittend place
        self.current_index = (self.current_index + 1) % self.history_length

    def get_obs_vec(self, obs_ids):
        sorted_obs_ids = sorted(obs_ids,reverse=True)

        # get the real index according to the pointer
        indices = (self.current_index - 1 - torch.tensor(sorted_obs_ids, device=self.device)) % self.history_length
        
        expanded_indices = indices.expand(self.num_envs, self.num_obs, -1)
        selected = torch.gather(self.obs_history, 2, expanded_indices)
        
        # (env, obs, time) -> (env, time, obs)
        selected = selected.permute(0, 2, 1)
        # reshape to (num_envs, num_obs * len(obs_ids))
        return selected.reshape(self.num_envs, -1)
