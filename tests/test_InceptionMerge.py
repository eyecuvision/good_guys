from unittest import TestCase
import torch
from good_guys.layers import InceptionMerge


class TestInceptionMerge(TestCase):
    def test_forward(self):
        FILTER_SIZE = 4
        DIM = 16

        # Single entity test
        BATCH_SIZE = 1

        shape = (BATCH_SIZE, FILTER_SIZE, DIM, DIM)
        inp = torch.randn((shape))

        inception = InceptionMerge(FILTER_SIZE)
        res = inception(inp)
        expected_shape = (BATCH_SIZE, 1, DIM//2, DIM//2)
        self.assertEqual(res.shape, expected_shape)

        # Minibatch
        BATCH_SIZE = 16

        shape = (BATCH_SIZE, FILTER_SIZE, DIM, DIM)
        inp = torch.randn((shape))

        inception = InceptionMerge(FILTER_SIZE)
        res = inception(inp)
        expected_shape = (BATCH_SIZE, 1, DIM//2, DIM//2)
        self.assertEqual(res.shape, expected_shape)
