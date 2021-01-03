import torch.nn as N
import torch

class FPN(N.Module):


    def __init__(self,filters : [int],stride = 2):
        super().__init__()


        self.convs = [N.Conv2d(in_channel,in_channel,1) for in_channel in reversed(filters)]
        self.upsamples = [None] + [N.ConvTranspose2d(in_channel,out_channel,4,stride,1) for in_channel,out_channel in reversed(list(zip(filters[1:],filters[:-1])))]


    def forward(self,features : [torch.Tensor]):

        results = []
        prev = None

        for x,conv,upsample in zip(reversed(features),self.convs,self.upsamples):

            if upsample is None:
                prev = x
            else:
                prev = upsample(prev)

            x = conv(x)
            prev = x + prev
            results.append(prev)

        return reversed(results)
