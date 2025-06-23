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

def test2():
    # speed test across three implementations
    import time
    num_envs = 4096
    num_obs = 12
    include_history_steps = 40
    device = torch.device("cuda")
    large_buffer1 = ObservationBuffer(num_envs, num_obs, include_history_steps, device)
    large_buffer2 = ObservationBuffer3D(num_envs, num_obs, include_history_steps, device)
    large_buffer3 = ObservationBuffer3D_circular(num_envs, num_obs, include_history_steps, device)

    test_iter = 10000
    obs = torch.rand(num_envs, num_obs, test_iter, device=device)

    large_buffer1.insert(obs[...,0])
    start = time.time()
    for it in range(test_iter):
        large_buffer1.insert(obs[...,it])
    end = time.time()
    print(f"{large_buffer1.__class__} insert {test_iter} using: {end-start:.4f} seconds")

    large_buffer2.insert(obs[...,0])
    start = time.time()
    for it in range(test_iter):
        large_buffer2.insert(obs[...,it])
    end = time.time()
    print(f"{large_buffer2.__class__} insert {test_iter} using: {end-start:.4f} seconds")

    large_buffer3.insert(obs[...,0])
    start = time.time()
    for it in range(test_iter):
        large_buffer3.insert(obs[...,it])
    end = time.time()
    print(f"{large_buffer3.__class__} insert {test_iter} using: {end-start:.4f} seconds")


    r1 = large_buffer1.get_obs_vec(np.arange(include_history_steps))
    r2 = large_buffer2.get_obs_vec(np.arange(include_history_steps))
    r3 = large_buffer3.get_obs_vec(np.arange(include_history_steps))
    diff2 = r2 - r1
    diff3 = r3 - r1
    if (diff2!=0).sum() >0:
        print(it, (diff2!=0).sum())
        print((diff2!=0).nonzero())
        return
    if (diff3!=0).sum() >0:
        print(it, (diff3!=0).sum())
        print((diff3!=0).nonzero())
        return
    print(diff2.mean().item(),diff3.mean().item())

def test3():
    # larget batch error test
    num_envs = 400
    num_obs = 12
    include_history_steps = 40
    device = torch.device("cuda")
    large_buffer1 = ObservationBuffer(num_envs, num_obs, include_history_steps, device)
    large_buffer2 = ObservationBuffer3D(num_envs, num_obs, include_history_steps, device)
    large_buffer3 = ObservationBuffer3D_circular(num_envs, num_obs, include_history_steps, device)

    test_iter = 10000
    obs = torch.rand(num_envs, num_obs, test_iter, device=device)
    obs_r = torch.zeros(num_envs, num_obs, device=device)

    for it in range(test_iter):
        large_buffer1.insert(obs[...,it])
        large_buffer2.insert(obs[...,it])
        large_buffer3.insert(obs[...,it])
        r1 = large_buffer1.get_obs_vec(np.arange(include_history_steps))
        r2 = large_buffer2.get_obs_vec(np.arange(include_history_steps))
        r3 = large_buffer3.get_obs_vec(np.arange(include_history_steps))
        diff2 = r2 - r1
        diff3 = r3 - r1

        if it % 100 == 0:
            large_buffer1.reset(torch.arange(num_envs,device=device),obs_r)
            large_buffer2.reset(torch.arange(num_envs,device=device),obs_r)
            large_buffer3.reset(torch.arange(num_envs,device=device),obs_r)
        
        if (diff2!=0).sum() >0:
            print(it, (diff2!=0).sum())
            print((diff2!=0).nonzero())
            return

        if (diff3!=0).sum() >0:
            print(it, (diff3!=0).sum())
            print((diff3!=0).nonzero())
            return
    print(f"larget batch error test: ok")

if __name__ == "__main__":
    test2()
    test3()