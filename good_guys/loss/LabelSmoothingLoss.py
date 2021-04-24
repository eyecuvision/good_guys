import torch.nn as N
import torch

class LabelSmoothingLoss(N.Module):
    def __init__(self, number_of_classes, smoothing=0,temperature = 1, dim=-1, weight = None):

        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.number_of_classes = number_of_classes
        self.dim = dim
        self.temperature = temperature

        assert 0 <= self.smoothing < 1


    def forward(self, pred, target):
        pred = torch.log_softmax(pred/self.temperature,dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.number_of_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


