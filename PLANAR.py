
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
from utils import ACTIVATION_DERIVATIVES
from flow_utils import *
from torch.distributions.multivariate_normal import MultivariateNormal


class PlanarFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh):
        super().__init__()
        self.D = D
        self.w = nn.Parameter(torch.empty(D))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(D))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def newton_method(function, initial, iteration=10, convergence=0.0001):
        for i in range(iteration):
            previous_data = initial.clone()
            value = function(initial)
            value.backward()
            # update
            initial.data -= (value / initial.grad).data
            # zero out current gradient to hold new gradients in next iteration
            initial.grad.data.zero_()
            print("epoch {}, obtain {}".format(i, initial))
            # Check convergence.
            # When difference current epoch result and previous one is less than
            # convergence factor, return result.
            if torch.abs(initial - previous_data) < torch.tensor(convergence):
                print("break")
                return initial.data
        return initial.data  # return our final after iteration

    def forward(self, x: torch.Tensor):
        lin = (x @ self.w + self.b).unsqueeze(1)  # shape: (B, 1)
        f = x + self.u * self.activation(lin)  # shape: (B, D)
        phi = self.activation_derivative(lin) * self.w  # shape: (B, D)
        log_det = torch.log(torch.abs(1 + phi @ self.u) + 1e-4)  # shape: (B,)

        return f, log_det

    def inverse(self, z: torch.Tensor):




flow_planar = NVP_flow([PlanarFlow(2)]*10)

print(flow_planar)


# function to want to solve
def solve_func(x):
    return torch.exp(x) - 2



# call starts
result = newton_method(solve_func, initial_x)



