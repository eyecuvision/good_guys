import torch
import torch.nn as N


class Inception(N.Module):

    def __init__(self, in_filters, out_filters):

        super().__init__()
        self.x1 = N.Conv2d(in_filters,out_filters//4,1,2)
        self.x2 = N.Sequential(
            N.Conv2d(in_filters, in_filters, 1),
            N.Conv2d(in_filters, out_filters // 4, 3, 2,1),
        )
        self.x3 = N.Sequential(
            N.Conv2d(in_filters, in_filters, 1),
            N.Conv2d(in_filters, out_filters // 4, 5, 2,2),
        )
        self.x4 = N.MaxPool2d(3,2,1)


    def forward(self,x):

        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        x4 = self.x4(x)

        return torch.cat([x1,x2,x3,x4],dim=1)