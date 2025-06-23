# created by Xu Xin, 2025
import torch

class ObservationBuffer3D:
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

    def reset(self, reset_ids, new_obs):
        self.obs_history[reset_ids,:,:] = new_obs.unsqueeze(-1).repeat(1, 1, self.history_length)

    def insert(self, new_obs):
        self.obs_history[:, :, :-1] = self.obs_history[:, :, 1:].clone()
        self.obs_history[:, :, -1].copy_(new_obs)

    def get_obs_vec(self, obs_ids):
        """Gets history of observations indexed by obs_ids.
        
        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        """
        obs = []
        for obs_id in reversed(sorted(obs_ids)):
            slice_idx = self.history_length - obs_id - 1
            obs.append(self.obs_history[:, :, slice_idx])
        return torch.cat(obs, dim=-1)

