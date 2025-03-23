import torch
import os
from glob import glob
import numpy as np
from torch.nn import functional as F
import time

class Generator(object):
    def __init__(self, model, exp_path, res=256, threshold = 0.1, device=torch.device("cuda:0")):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.exp_path = exp_path
        self.load_checkpoint()
        self.threshold = threshold
        self.res = res

    def generate_point_cloud(self, data, num_steps = 10, num_points = 10000, filter_val = 0.009):
        start = time.time()
        data = np.unpackbits(data)
        inputs = torch.tensor(np.reshape(np.array(data, dtype=np.float32), (self.res,) * 3)).unsqueeze(0).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        '''
        sample_num is set according to the size of data block.
        The size of test sample data block here is twice that of the training sample block in our experiment.
        '''
        sample_num = 400000
        #sample_num = 200000
        samples_cpu = np.zeros((0, 3))
        samples = torch.rand(1, sample_num, 3).float().to(self.device) * 3 - 1.5
        samples.requires_grad = True
        encoding = self.model.encoder(inputs)
        i = 0
        while len(samples_cpu) < num_points:
            print('iteration', i)
            for j in range(num_steps):
                print('refinement', j)
                df_pred = torch.clamp(self.model.decoder(samples, *encoding), max=self.threshold)
                df_pred.sum().backward()
                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                inputs = inputs.detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)
                samples = samples.detach()
                samples.requires_grad = True
            print('finished refinement')
            if not i == 0:
                samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))
            samples = samples[df_pred < 0.03].unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * torch.randn(samples.shape).to(self.device)
            samples = samples.detach()
            samples.requires_grad = True
            i += 1
            print(samples_cpu.shape)
        duration = time.time() - start
        return samples_cpu, duration


    def load_checkpoint(self):
        checkpoints = glob(self.exp_path + '*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.exp_path))
            return 0, 0
        else:
            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.exp_path + 'checkpoint_{}.tar'.format(checkpoints[-1])
            print('Loaded checkpoint from: {}'.format(path))
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            training_time = checkpoint['training_time']
            return epoch, training_time


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds
