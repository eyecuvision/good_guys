import torch.nn as N
import torch

from .DepthwiseSeparableConv2d import DepthwiseSeparableConv2d


class TemporalDepthwiseSeparableConv2d(N.Module):

    def __init__(self, filters: int, kernel_size: int):

        super().__init__()

        self.conv = DepthwiseSeparableConv2d(filters,kernel_size)

    def forward(self,x):
        result = self.conv(x)
        result = result.view(1, *result.shape)
        return torch.sum(result, dim=1)

