from typing import Tuple

import torch.nn as N
import torch


class ResizeTopDown(N.Module):

    def __init__(self, in_filter: int, out_filter: int, kernel_size: int, stride: int = 2):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.l = N.Conv2d(in_filter, out_filter, kernel_size, stride, padding, groups=in_filter)

    def forward(self, x):
        return self.l(x)


class ResizeBottomUp(N.Module):

    def __init__(self, in_filter: int, out_filter: int, kernel_size: int, stride: int = 2):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.l = N.ConvTranspose2d(in_filter, out_filter, kernel_size, stride, padding, groups=out_filter)

    def forward(self, x):
        return self.l(x)


class ConvNormLeakyRelu(N.Module):

    def __init__(self, conv_module: N.Module, out_filter: int):
        super().__init__()

        self.l = N.Sequential(
            conv_module,
            N.BatchNorm2d(out_filter),
            N.LeakyReLU(inplace=True)
        )

    def forward(self,x):

        return self.l(x)


class BiFPN(N.Module):

    def __init__(self, filters: [int], eps: float = 10e-6):
        super().__init__()
        self.eps = eps
        self.filters = filters
        self.number_of_inputs = len(filters)

        self.down_weights = [
            2 * (N.Parameter(data=torch.randn(1)),) for _ in range(self.number_of_inputs - 1)
        ]

        self.up_weights = [
            3 * (N.Parameter(data=torch.randn(1)),) for _ in range(self.number_of_inputs - 1)
        ]

        self.downsample = [
            ConvNormLeakyRelu(ResizeTopDown(in_filter, out_filter, 5),out_filter) for in_filter, out_filter in
            zip(filters[:-1], filters[1:])
        ]
        self.upsample = [
            ConvNormLeakyRelu(ResizeBottomUp(in_filter, out_filter, 4),out_filter) for in_filter, out_filter in
            zip(filters[1:], filters[:-1])
        ]

        self.td_convs = [
            ConvNormLeakyRelu(N.Conv2d(filter,filter, 5, 1, 2,groups=filter, bias=False),filter) for filter in self.filters
        ]

        self.out_convs = [
            ConvNormLeakyRelu(N.Conv2d(filter, filter, 5, 1, 2,groups=filter, bias=False),filter) for filter in self.filters
        ]

    def fast_fusion_top_down(self, x):

        td_list = []

        for ind, (down_1, down_2) in reversed(list(enumerate(self.down_weights))):
            tmp_1 = down_1 * self.upsample[ind](x[ind + 1])
            tmp_2 = down_2 * x[ind]
            result = self.td_convs[ind]((tmp_1 + tmp_2) *
                                        1 / (self.eps + down_1.data.squeeze() + down_2.data.squeeze()))
            td_list.append(result)

        return list(reversed(td_list))

    def fast_fusion_out(self, x, td_list):

        out_list = [self.out_convs[0](td_list.pop(0))]

        for ind, (up_1, up_2, up_3) in enumerate(self.up_weights):

            tmp_1 = up_1 * x[ind + 1]
            tmp_3 = up_3 * self.downsample[ind](out_list[-1])

            if len(td_list):
                tmp_2 = up_2 * td_list.pop(0)
                result = self.out_convs[ind + 1]((tmp_1 + tmp_2 + tmp_3) *
                                             1 / (self.eps + up_1.data.squeeze() + up_2.data.squeeze() + up_3.data.squeeze()))
            else:
                result = self.out_convs[ind + 1]((tmp_1 + tmp_3) *
                                             1 / (self.eps + up_1.data.squeeze() + up_3.data.squeeze()))

            out_list.append(result)

        return out_list

    def forward(self, x) -> Tuple[torch.Tensor]:

        assert len(x) == self.number_of_inputs, f"Expected input shape (batch_size,{self.number_of_inputs},h,w)."

        td_list = self.fast_fusion_top_down(x)
        out_list = self.fast_fusion_out(x, td_list)

        return tuple(out_list)
