import torch.nn as N


class Downsample1d(N.Module):

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x[:, :, ::self.scale_factor]
