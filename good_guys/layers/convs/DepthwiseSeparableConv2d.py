import torch.nn as N
from typing import Union, Tuple

from good_guys.utils.double import double


class DepthwiseSeparableConv2d(N.Module):

    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1):
        super().__init__()

        kernel_size = double(kernel_size)
        stride = double(stride)

        padding_1 = (kernel_size[0] - 1) // 2
        padding_2 = (kernel_size[1] - 1) // 2

        self.l = N.Sequential(
            N.Conv2d(filters, filters, (kernel_size[0], 1), (stride[0], 1), (padding_1, 0), groups=filters),
            N.Conv2d(filters, filters, (1, kernel_size[1]), (1, stride[1]), (0, padding_2), groups=filters)
        )

    def forward(self, x):
        return self.l(x)
