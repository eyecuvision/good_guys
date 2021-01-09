import torch.nn as N
from good_guys.activations import Mish


class ResBottleneck(N.Module):

    def __init__(self, submodule: N.Module,filters:int):
        super().__init__()
        self.module = N.Sequential(
            N.Conv2d(filters,filters,1),
            Mish(),
            submodule,
            N.Conv2d(filters,filters,1),
            Mish(),
        )

        self.out = Mish()

    def forward(self,x):

        x = self.module(x) + x
        return self.out(x)


