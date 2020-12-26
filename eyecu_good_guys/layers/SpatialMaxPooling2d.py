import torch.nn as N

class SpatialMaxPooling(N.Module):

    def forward(self,x):

        max_tensor,_ = x.max(dim = 1)
        return max_tensor.unsqueeze(1)