import torch.nn as N


class Downsample2d(N.Module):

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x[:, :, ::self.scale_factor, ::self.scale_factor]
