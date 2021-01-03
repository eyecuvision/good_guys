from unittest import TestCase
from good_guys.layers.pooling import BiFPN
import torch


class TestBiFPN(TestCase):
    BATCH_SIZES = [1, 16]
    FILTERS = [16, 32, 64, 128, 128]

    def test_forward_divisible_by_2(self):

        # Bigger images have smaller indices

        for batch_size in self.BATCH_SIZES:

            root_dim = 64
            dim = root_dim

            inputs = []

            for filter in self.FILTERS:
                image = torch.randn((batch_size, filter, dim, dim))
                inputs.append(image)
                dim = dim // 2

            module = BiFPN(self.FILTERS)
            result =  module(inputs)

            for inp,result in zip(inputs,result):

                expected_size = list(inp.size())
                result_size   = list(result.size())

                self.assertEqual(expected_size,result_size)


    def test_forward_not_divisible_by_2(self):

        pass
