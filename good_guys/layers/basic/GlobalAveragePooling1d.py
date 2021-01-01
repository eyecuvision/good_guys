import torch.nn as N

class GlobalAveragePooling1d(N.Module):


    def forward(self,x):

        return x.mean([2])