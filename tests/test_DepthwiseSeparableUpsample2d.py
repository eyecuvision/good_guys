from unittest import TestCase
from good_guys.layers import DepthwiseSeparableUpsample2d
import torch

class TestUpsampleBlock(TestCase):
    def test_stride(self):
        batch_size = 1
        in_filter = 8
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableUpsample2d(in_filter,in_filter, 6)

        inp = torch.randn(shape)

        result = layer(inp)

        expected_shape = (batch_size, in_filter, dim*2, dim*2)
        self.assertEqual(expected_shape, result.shape)

        batch_size = 32
        in_filter = 1
        dim = 28

        shape = (batch_size, in_filter, dim, dim)

        layer = DepthwiseSeparableUpsample2d(in_filter,in_filter, (6, 4))

        inp = torch.randn(shape)

        result = layer(inp)
        expected_shape = (batch_size, in_filter, dim*2, dim*2)

        self.assertEqual(expected_shape, result.shape, )




