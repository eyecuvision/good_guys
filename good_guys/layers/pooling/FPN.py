import torch.nn as N
import torch

from good_guys.layers.misc import AttrProxy


class FPN(N.Module):

    def __init__(self, filters: [int], stride=2):
        super().__init__()

        self.convs = AttrProxy(self, "convs_")
        self.upsamples = AttrProxy(self, "upsamples_")

        for ind, in_channel in enumerate(reversed(filters)):
            self.add_module(
                self.convs(ind),
                N.Conv2d(in_channel, in_channel, 1)
            )

        self.add_module(
            self.upsamples(0),
            N.Identity()
        )
        for ind, (in_channel, out_channel) in enumerate(reversed(list(zip(filters[1:], filters[:-1])))):
            self.add_module(
                self.upsamples(ind + 1),
                N.ConvTranspose2d(in_channel, out_channel, 4, stride, 1)
            )

    def forward(self, features: [torch.Tensor]):

        results = []
        prev = features[-1]

        for ind, x in enumerate(reversed(features)):
            up_module = self.upsamples[ind]
            conv_module = self.convs[ind]

            prev = up_module(prev)

            x = conv_module(x)
            prev = x + prev
            results.append(prev)

        return reversed(results)
