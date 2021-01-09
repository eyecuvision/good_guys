from unittest import TestCase
from good_guys.layers import ResBottleneck
import torch.nn as N
import torch


class TestResBottleneck(TestCase):
    def test_forward(self):

        filters = 64
        dim = 16

        submodule = N.Conv2d(filters, filters, 1)
        layer = ResBottleneck(submodule, filters)

        batch_size = 1




        shape = batch_size,filters,dim,dim

        inp = torch.randn(shape)
        result = layer(inp)

        self.assertEqual(shape,result.shape)

        batch_size = 32
        shape = batch_size,filters,dim,dim


        inp = torch.randn(shape)
        result = layer(inp)
        self.assertEqual(shape,result.shape)
