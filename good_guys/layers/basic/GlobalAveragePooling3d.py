import torch.nn as N

class GlobalAveragePooling3d(N.Module):


    def forward(self,x):

        return x.mean([2,3,4])