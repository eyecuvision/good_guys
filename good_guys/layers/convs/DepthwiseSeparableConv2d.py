import torch.nn as N
from good_guys.activations import Mish


class DepthwiseSeparableConv2d(N.Module):

    def __init__(self,filters : int,kernel_size:int):
        super().__init__()

        self.padding = (kernel_size-1)//2

        self.l = N.Sequential(
            N.Conv2d(filters,filters,(kernel_size,1),1,(self.padding,0),groups=filters),
            N.BatchNorm2d(filters),
            Mish(),
            N.Conv2d(filters,filters,(1,kernel_size),1,(0,self.padding),groups=filters),
            N.BatchNorm2d(filters),
            Mish(),
        )



    def forward(self,x):

        return self.l(x)
