import torch.nn as N
import torch


class Mish(N.Module):

    def forward(self, x):
        x = x * (torch.tanh(N.functional.softplus(x)))
        return x
