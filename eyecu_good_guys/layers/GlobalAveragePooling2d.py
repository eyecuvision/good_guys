import torch.nn as N

class GlobalAveragePooling2d(N.Module):


    def forward(self,x):

        return x.mean([2,3])