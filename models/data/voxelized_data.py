from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch

class VoxelizedDataset(Dataset):
    def __init__(self, mode, res, split_file, batch_size, num_workers, sample_distribution, sample_sigmas):
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)
        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)
        self.split = np.load(split_file)
        self.mode = mode
        self.data = self.split[mode]
        self.res = res
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_path = self.data[idx]
        occupancies = np.unpackbits(np.load(sample_path)['compressed_occupancies'])
        input_voxel = np.reshape(occupancies, (self.res,)*3)
        raw_points = np.load(sample_path)['point_cloud']
        return {'inputs': np.array(input_voxel, dtype=np.float32), 'points': raw_points}

    def get_loader(self, shuffle =True):
        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
