from unittest import TestCase

import torch

from good_guys.layers import  DepthwiseSeparableConv3d


class TestDepthwiseSeparableConv3d(TestCase):
    def test_stride(self):
        batch_size = 1
        in_filter = 64
        out_filter = 64
        dim = 28
        kernel_size = 15
        stride = 4

        layer = DepthwiseSeparableConv3d(in_filter, kernel_size, stride)

        inp = torch.randn(batch_size, in_filter,dim, dim, dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim // stride, dim // stride,dim // stride)

        self.assertEqual(expected_shape, result.shape)

        batch_size = 64
        stride = 2

        layer = DepthwiseSeparableConv3d(in_filter, kernel_size, (stride, 1,stride))

        inp = torch.randn(batch_size, in_filter, dim, dim,dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim // stride, dim,dim // stride)

        self.assertEqual(expected_shape, result.shape)

        layer = DepthwiseSeparableConv3d(in_filter, kernel_size, (1, stride,1))

        inp = torch.randn(batch_size, in_filter, dim, dim,dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim, dim // stride,dim)

        self.assertEqual(expected_shape, result.shape)


    def test_kernel_size(self):
        batch_size = 1
        in_filter = 8
        dim = 28

        shape = (batch_size, in_filter, dim, dim,dim)

        layer = DepthwiseSeparableConv3d(in_filter, (3,5,7))

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape, result.shape)

        batch_size = 32
        in_filter = 1
        dim = 28

        shape = (batch_size, in_filter, dim, dim,dim)

        layer = DepthwiseSeparableConv3d(in_filter, (7, 5,5))

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape, result.shape, )

        batch_size = 32
        in_filter = 32
        dim = 28

        shape = (batch_size, in_filter, dim, dim,dim)

        layer = DepthwiseSeparableConv3d(in_filter, 7)

        inp = torch.randn(shape)

        result = layer(inp)

        self.assertEqual(shape, result.shape)
