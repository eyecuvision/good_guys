from unittest import TestCase
from good_guys.layers import Inception
import torch



class TestInception(TestCase):
    def test_forward(self):


        FILTER_SIZE = 4
        DIM = 16

        #Single batch training
        shape = (1,FILTER_SIZE,DIM,16)
        inp = torch.randn((shape))

        inception = Inception(FILTER_SIZE,FILTER_SIZE)
        res = inception(inp)
        expected_shape = shape
        self.assertEqual(res.shape, expected_shape)

        shape = (1, FILTER_SIZE, DIM//2, DIM//2)
        inp = torch.randn((shape))

        inception = Inception(FILTER_SIZE, FILTER_SIZE*4)
        res = inception(inp)
        expected_shape = (1,FILTER_SIZE*4,DIM//2,DIM//2)
        self.assertEqual(res.shape, expected_shape)


        BATCH_SIZE = 16
        shape = (BATCH_SIZE, FILTER_SIZE, DIM, DIM)
        inp = torch.randn((shape))

        inception = Inception(FILTER_SIZE, FILTER_SIZE * 4)
        res = inception(inp)
        expected_shape = (BATCH_SIZE, FILTER_SIZE * 4, DIM//2, 16//2)
        self.assertEqual(res.shape, expected_shape)
