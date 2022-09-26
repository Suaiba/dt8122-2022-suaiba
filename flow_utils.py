from datasets import *
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from train_test_split import *
from planar_flow import *
from planar_transfrom import *
from distributions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import glob
from torch.distributions import Normal
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq  import odeint
from hyper_net import *



class Dataset2D(Dataset):
    def __init__(self, csv_file, transform=None):
        """
              Args:
                  csv_file (string): Path to the csv file with annotations.
                  transform (callable, optional): Optional transform to be applied
                      on a sample.
              """
        super().__init__()
        self.fulldata_frame = pd.read_csv(csv_file)



    def __len__(self):
        # implement len() function, to find total number of data
        return len(self.fulldata_frame)

    def __getitem__(self, idx):
        x = (self.fulldata_frame.iloc[idx,0])
        y = (self.fulldata_frame.iloc[idx,1])
        data= torch.tensor([x,y])

        return x, y, data


def get_train_test_loader(csvfilename):
    _data = Dataset2D(csv_file=csvfilename)
    N = len(_data)
    indices = np.random.RandomState(seed=10).permutation(np.arange(0, N))
    indices_train = indices[0:int(N * 0.8)]
    indices_test = indices[int(N * 0.8):]
    train_sampler, test_sampler = SubsetRandomSampler(indices_train), SubsetRandomSampler(indices_test)
    train_dataloader = DataLoader(_data, batch_size=32, num_workers=0, sampler=train_sampler)
    test_dataloader = DataLoader(_data, batch_size=32, num_workers=0, sampler=test_sampler)
    return train_dataloader, test_dataloader



class NVP_flow(nn.Module):

    def __init__(self, bijections):
        super().__init__()
        self.bijections = nn.ModuleList(bijections)

    @property
    def base_dist(self):
        return Normal(
            loc=torch.zeros(2),
            scale=torch.ones(2),
        )

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0])
        for bijection in self.bijections:
            x, ldj = bijection(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x).sum(1)
        # print("log:",log_prob)
        return log_prob

    def prob(self, x):
        # log_prob = torch.zeros(x.shape[0])
        for bijection in self.bijections:
            x,_ = bijection(x)
            # log_prob += ldj
        # print("x1:",x)
        # log_prob += self.base_dist.log_prob(x).sum(1)
        return x

    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        for bijection in reversed(self.bijections):
            z = bijection.inverse(z)
        return z


class ReverseBijection(nn.Module):

    def forward(self, x):
        # print("x:",x.type())
        return x.flip(-1), x.new_zeros(x.shape[0])

    def inverse(self, z):
        return z.flip(-1)


class CouplingBijection(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        id, x2 = torch.chunk(x, 2, dim=-1)
        p = self.net(id)
        log_s, b = torch.chunk(p, 2, dim=-1)
        z2 = x2 * log_s.exp() + b
        z = torch.cat([id, z2], dim=-1)
        ldj = log_s.sum(-1)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            id, z2 = torch.chunk(z, 2, dim=-1)
            p = self.net(id)
            log_s, b = torch.chunk(p, 2, dim=-1)
            x2 = (z2 - b) * (-log_s).exp()
            x = torch.cat([id, x2], dim=-1)
        return x






ACTIVATION_DERIVATIVES = {
    F.elu: lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
    torch.tanh: lambda x: 1 - torch.tanh(x) ** 2
}




class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self,in_out_dim, hidden_dim, width ):
        super().__init__()
        self.net = HyperNetwork(in_out_dim, hidden_dim, width)

    def trace_df_dz(self, f, z):
        """Calculates the trace of the Jacobian df/dz.
        Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
        Input:
            f - function output [N,d]
            z - current state [N,d]
        Returns:
            tr(df/dz) - [N]
        """
        sum_diag = 0.
        for i in range(z.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
        return sum_diag.contiguous()

    def ode_rhs(self, t, states):
        ''' Differential function implementation. states is (x1,logp_diff_t1) where
                x1 - [N,d] initial values for ODE states
                logp_diff_t1 - [N,1] initial values for density changes
        '''
        z, logp_z = states  # [N,d], [N,1]
        N = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.net(t, z)  # [N,d]
            dlogp_z_dt = -self.trace_df_dz(dz_dt, z).view(N, 1)
        return (dz_dt, dlogp_z_dt)

    def forward(self, ts, z0, logp_diff_t0, method='dopri5'):
        ''' Forward integrates the CNF system. Returns state and density change solutions.
            Input
                ts - [T]   time points
                z0 - [N,d] initial values for ODE states
                logp_diff_t0 - [N,1] initial values for density changes
            Retuns:
                zt -     [T,N,...]  state trajectory computed at t
                logp_t - [T,N,1]    density change computed over time
        '''
        zt, logp_t = odeint(self.ode_rhs, (z0, logp_diff_t0), ts, method=method)
        return zt, logp_t


def prior():
    return Normal(
        loc=torch.zeros(2),
        scale=torch.ones(2),
    )
