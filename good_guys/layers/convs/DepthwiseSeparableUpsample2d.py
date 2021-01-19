import torch.nn as N

from good_guys.utils import double


class DepthwiseSeparableUpsample2d(N.Module):

    def __init__(self,in_filter,out_filter,kernel_size,stride = 2):

        kernel_size = double(kernel_size)
        stride = double(stride)
        padding = (kernel_size[0]-1 ) // 2, (kernel_size[1]-1 )//2

        super().__init__()
        self.l = N.Sequential(
            N.Conv2d(in_filter, out_filter, 1),
            N.ConvTranspose2d(out_filter,out_filter,(kernel_size[0],1),(stride[0],1),(padding[0],0),groups=out_filter),
            N.ConvTranspose2d(out_filter,out_filter,(1,kernel_size[1]),(1,stride[1]),(0,padding[1]),groups=out_filter),
            N.Conv2d(out_filter, out_filter, 1)
        )

    def forward(self,x):

        return self.l(x)