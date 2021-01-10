from unittest import TestCase
from good_guys.layers import DepthwiseSeparableConv2d
import torch


class TestDepthwiseSeperableConv2d(TestCase):
    def test_kernel_size(self):
        batch_size = 1
        in_filter = 8
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConv2d(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape, result.shape)

        batch_size = 32
        in_filter = 1
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConv2d(in_filter, (7, 5))

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape, result.shape, )

        batch_size = 32
        in_filter = 32
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableConv2d(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape, result.shape)

    def test_filter_shapes(self):

        channels = 64
        kernel = 7

        layer = DepthwiseSeparableConv2d(channels, kernel)
        for params in layer.parameters():

            if len(params.shape) == 4:
                self.assertEqual(64 * 7, torch.prod(torch.Tensor([params.shape])))

    def test_stride(self):

        batch_size = 1
        in_filter = 64
        out_filter = 64
        dim = 28
        kernel_size = 15
        stride = 4

        layer = DepthwiseSeparableConv2d(in_filter, kernel_size, stride)

        inp = torch.randn(batch_size, in_filter, dim, dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim // stride, dim // stride)

        self.assertEqual(expected_shape, result.shape)

        batch_size = 64
        stride = 2

        layer = DepthwiseSeparableConv2d(in_filter, kernel_size, (stride, 1))

        inp = torch.randn(batch_size, in_filter, dim, dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim // stride, dim)

        self.assertEqual(expected_shape, result.shape)

        layer = DepthwiseSeparableConv2d(in_filter, kernel_size, (1, stride))

        inp = torch.randn(batch_size, in_filter, dim, dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim, dim // stride)

        self.assertEqual(expected_shape, result.shape)
