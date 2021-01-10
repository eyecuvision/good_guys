import torch.nn as N
from typing import Union, Tuple


def triple(k) -> (int, int,int):
    if type(k) == int:
        return k, k,k
    else:
        return k


class DepthwiseSeparableConv3d(N.Module):

    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int, int]], stride : Union[int, Tuple[int, int, int]] =1):
        super().__init__()

        kernel_size = triple(kernel_size)
        stride = triple(stride)

        padding_1 = (kernel_size[0] - 1) // 2
        padding_2 = (kernel_size[1] - 1) // 2
        padding_3 = (kernel_size[2] - 1) // 2

        self.l = N.Sequential(
            N.Conv3d(filters, filters, (kernel_size[0], 1, 1), (stride[0], 1, 1), (padding_1, 0, 0), groups=filters),
            N.Conv3d(filters, filters, (1, kernel_size[1],1), (1, stride[1], 1), (0, padding_2, 0), groups=filters),
            N.Conv3d(filters, filters, (1, 1,kernel_size[2]), (1, 1, stride[2]), (0, 0, padding_3),groups=filters)
        )

    def forward(self, x):
        return self.l(x)
