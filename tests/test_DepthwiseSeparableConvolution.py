from unittest import TestCase
from good_guys.layers import DepthwiseSeparableConvolution
import torch


class TestDepthwiseSeperableConvolution(TestCase):
    def test_forward(self):

        batch_size = 1
        in_filter = 8
        dim = 28

        shape = (batch_size,in_filter,dim,dim)

        layer = DepthwiseSeparableConvolution(in_filter,7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(result.shape,shape)

        batch_size = 32
        in_filter = 1
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConvolution(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(result.shape, shape)


        batch_size = 32
        in_filter = 32
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConvolution(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(result.shape, shape)