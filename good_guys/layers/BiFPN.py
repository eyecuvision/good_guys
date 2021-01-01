from typing import Tuple

import torch.nn as N
import torch


class BiFPN(N.Module):

    def __init__(self, in_channels, out_channels, number_of_inputs: int, eps: float = 10e-6):
        super().__init__()
        self.number_of_inputs = number_of_inputs
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down_weights = [
            (N.Conv2d(in_channels, in_channels, 1, bias=False), N.Conv2d(in_channels, in_channels, 1, bias=False)) for _
            in range(self.number_of_inputs - 1)
        ]

        self.up_weights = [
            (N.Conv2d(in_channels, in_channels, 1, bias=False), N.Conv2d(in_channels, in_channels, 1, bias=False)) for _
            in range(self.number_of_inputs - 1)
        ]

        self.downsample = [
            N.AvgPool2d(2, 2) for _ in range(self.number_of_inputs - 1)
        ]

        self.upsample = [
            N.UpsamplingBilinear2d(scale_factor=2) for _ in range(self.number_of_inputs - 1)
        ]

        self.convs_1 = [
            N.Conv2d(in_channels, in_channels, 5, 1, 2, bias=False) for _ in range(self.number_of_inputs - 1)
        ]

        self.convs_2 = [
            N.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False) for _ in range(self.number_of_inputs - 1)
        ]

    def forward(self, x) -> Tuple[torch.Tensor]:

        assert (x.shape[1] == self.number_of_inputs, f"Expected input shape (batch_size,{self.number_of_inputs},h,w).")

        tmp = []

        for ind, (down_1, down_2) in enumerate(self.down_weights):
            tmp_1 = down_1(self.downsample[ind](x[self.number_of_inputs - ind]))
            tmp_2 = down_2(x[self.number_of_inputs - ind - 1])
            result = self.convs_1[ind]((tmp_1 + tmp_2) *
                                       1 / (self.eps + down_1.weight.data.squeeze() + down_2.weight.data.squeeze()))
            tmp.append(result)

        results: [torch.Tensor] = []

        for ind, (up_1, up_2) in enumerate(reversed(self.up_weights)):
            tmp_1 = up_1(self.downsample[ind](x[self.number_of_inputs - ind]))
            tmp_2 = up_2(x[self.number_of_inputs - ind - 1])
            result = self.convs_2[ind]((tmp_1 + tmp_2) *
                                       1 / (self.eps + up_1.weight.data.squeeze() + up_2.weight.data.squeeze()))
            results.append(result)

        return tuple(*results)
