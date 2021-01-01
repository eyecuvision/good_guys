import torch.nn as N
from good_guys.layers.basic.SpatialMaxPooling import SpatialMaxPooling

class InceptionMerge(N.Module):

    def __init__(self, in_filters):

        super().__init__()
        self.x1 = N.Conv2d(in_filters,1,1,2)
        self.x2 = N.Sequential(
            N.Conv2d(in_filters, in_filters, 1),
            N.Conv2d(in_filters, 1, 3, 2,1),
        )
        self.x3 = N.Sequential(
            N.Conv2d(in_filters, in_filters, 1),
            N.Conv2d(in_filters, 1, 5, 2,2),
        )
        self.x4 = N.Sequential(
            SpatialMaxPooling(),
            N.MaxPool2d(2,2)
        )


    def forward(self,x):

        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        x4 = self.x4(x)

        return x1+x2+x3+x4
