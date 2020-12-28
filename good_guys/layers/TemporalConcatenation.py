import torch.nn as N
import torch


class TemporalConcatenation(N.Module):


    def forward(self,x):

        cat = torch.cat(x,dim=0)
        return cat.squeeze(0)