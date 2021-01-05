import torch.nn as N


class Reshape(N.Module):


    def __init__(self, shape,*args):
        super().__init__()
        if len(args) > 0:
            self.shape = [shape ,*args]
        else:
            self.shape = shape

    def forward(self,x):

        return x.view(x.shape[0],*self.shape)