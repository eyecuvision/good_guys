from unittest import TestCase
from good_guys.layers import DepthwiseSeparableConv2d
import torch


class TestDepthwiseSeperableConv2d(TestCase):
    def test_forward(self):
        batch_size = 1
        in_filter = 8
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConv2d(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape,result.shape, )

        batch_size = 32
        in_filter = 1
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConv2d(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual( shape,result.shape,)

        batch_size = 32
        in_filter = 32
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConv2d(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual( shape,result.shape)


    def test_filter_shapes(self):
        #TODO: Implement kernel weight size control.

        channels = 64
        kernel = 7

        layer = DepthwiseSeparableConv2d(channels, kernel)
        for params in layer.parameters():

            if len(params.shape) == 4:
                self.assertEqual(64*7,torch.prod(torch.Tensor([params.shape])))
