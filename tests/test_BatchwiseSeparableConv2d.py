from unittest import TestCase
from good_guys.layers import BatchwiseSeparableConv2d
import torch


class TestBatchwiseConv2d(TestCase):
    def test_forward(self):
        batch_size = 1
        in_filter = 64
        out_filter = 64
        dim = 28
        kernel_size = 15

        layer = BatchwiseSeparableConv2d(in_filter, kernel_size)

        inp = torch.randn(batch_size, in_filter, dim, dim)
        result = layer(inp)
        expected_shape = (batch_size, out_filter, dim, dim)

        self.assertEqual(expected_shape,result.shape )


        batch_size = 64

        inp = torch.randn(batch_size, in_filter, dim, dim)
        result = layer(inp)
        expected_shape = (1, out_filter, dim, dim)

        self.assertEqual(expected_shape,result.shape)

