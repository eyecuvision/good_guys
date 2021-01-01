import torch.nn as N


class DepthwiseSeparableConvolution(N.Module):

    def __init__(self,filters : int,kernel_size:int):
        super().__init__()
        self.l = N.Sequential(
            N.Conv2d(filters,filters,(kernel_size,1),1,(kernel_size//2,0),groups=filters),
            N.Conv2d(filters,filters,(1,kernel_size),1,(0,kernel_size//2),groups=filters),
            N.BatchNorm2d(filters),
            N.LeakyReLU(inplace=True),
            N.Conv2d(filters,filters,1),
            N.BatchNorm2d(filters),
            N.LeakyReLU(inplace=True)
        )



    def forward(self,x):

        return self.l(x)
