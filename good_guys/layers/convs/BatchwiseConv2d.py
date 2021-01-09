import torch
import torch.nn as N

from torch.nn.common_types import _size_2_t


class BatchwiseConv2d(N.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode)

    def forward(self, x):
        result = self.forward(x)
        result = result.view(1, *result.shape)
        return torch.sum(result, dim=1)
