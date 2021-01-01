from typing import Tuple

import torch.nn as N
import torch


class BiFPN(N.Module):

    def __init__(self, filters : int, number_of_inputs: int, eps: float = 10e-6):
        super().__init__()
        self.number_of_inputs = number_of_inputs
        self.eps = eps
        self.filters = filters

        self.down_weights = [
            2*(N.Parameter(data=torch.randn(1)),) for _ in range(self.number_of_inputs - 1)
        ]

        self.up_weights = [
            3*(N.Parameter(data=torch.randn(1)),) for _ in range(self.number_of_inputs - 1)
        ]

        self.downsample = N.AvgPool2d(2, 2)
        self.upsample = N.UpsamplingBilinear2d(scale_factor=2)


        self.td_convs = [
            N.Conv2d(self.filters, self.filters, 5, 1, 2, bias=False) for _ in range(self.number_of_inputs - 1)
        ]

        self.out_convs = [
            N.Conv2d(self.filters, self.filters, 5, 1, 2, bias=False) for _ in range(self.number_of_inputs - 1)
        ]


    def fast_fusion_top_down(self,x):
        
        
        td_list = []
        
        for ind, (down_1, down_2) in reversed(list(enumerate(self.down_weights))):
            tmp_1 = down_1 * self.upsample(x[ind+1])
            tmp_2 = down_2 * x[ind]
            result = self.td_convs[ind]((tmp_1 + tmp_2) *
                                       1 / (self.eps + down_1.data.squeeze() + down_2.data.squeeze()))
            td_list.append(result)
        
        return list(reversed(td_list))
    
    def fast_fusion_out(self,x,td_list):

        out_list = [td_list.pop(0)]

        for ind, (up_1, up_2,up_3) in enumerate(self.up_weights):

            tmp_1 = up_1 * x[ind + 1]
            tmp_3 = up_3 * self.downsample(out_list[-1])

            if len(td_list):
                tmp_2 = up_2 * td_list.pop(0)
                result = self.out_convs[ind]((tmp_1 + tmp_2 + tmp_3) *
                                             1 / (self.eps + up_1.data.squeeze() + up_2.data.squeeze() + up_3.data.squeeze()))
            else:
                result = self.out_convs[ind]((tmp_1 + tmp_3) *
                                             1 / (self.eps + up_1.data.squeeze()  + up_3.data.squeeze()))

            out_list.append(result)

        return out_list


    def forward(self, x) -> Tuple[torch.Tensor]:

        assert len(x) == self.number_of_inputs, f"Expected input shape (batch_size,{self.number_of_inputs},h,w)."

        td_list = self.fast_fusion_top_down(x)
        out_list = self.fast_fusion_out(x,td_list)


        return tuple(out_list)
