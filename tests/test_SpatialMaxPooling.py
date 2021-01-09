from unittest import TestCase
import torch
from good_guys.layers import SpatialMaxPooling


class TestSpatialMaxPooling(TestCase):
    def test_forward(self):

        pooling = SpatialMaxPooling()


        input_shape = (128,16,32,32)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (128,1,32,32)
        self.assertEqual(res.shape,result_shape)

        input_shape = (128, 1, 32, 32)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (128, 1, 32, 32)
        self.assertEqual(res.shape, result_shape)



        input_shape = (1, 128, 32, 32)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (1, 1, 32, 32)
        self.assertEqual(res.shape, result_shape)



        input_shape = (1, 1, 32, 32)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (1, 1, 32, 32)
        self.assertEqual(res.shape, result_shape)

        input_shape = (128, 64, 1, 32)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (128, 1, 1, 32)
        self.assertEqual(res.shape, result_shape)

        input_shape = (128, 64, 32, 1)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (128, 1, 32, 1)
        self.assertEqual(res.shape, result_shape)


        input_shape = (1, 1, 1, 1)
        arrrrr = torch.randn(input_shape)

        res = pooling(arrrrr)
        result_shape = (1, 1, 1, 1)
        self.assertEqual(res.shape, result_shape)


