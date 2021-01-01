from unittest import TestCase
from good_guys.layers import BiFPN
from itertools import product
import torch

class TestBiFPN(TestCase):
    IN_CHANNEL = [1,16,63]
    OUT_CHANNEL = [1,16,63]
    BATCH_SIZES = [1,16,63]
    LOG_2_VALUES = [4,5,6]

    def test_forward_divisible_by_2(self):

        # Bigger images have smaller indices


        for in_channel,out_channel,batch_size,log_2 in product(self.IN_CHANNEL,self.OUT_CHANNEL,self.BATCH_SIZES,self.LOG_2_VALUES):

            batch_size = 1
            layer = BiFPN(in_channel,out_channel,log_2)
            image = torch.randn((batch_size,in_channel,2**log_2,2**log_2))

            inputs = []
            for i in range(log_2):
                inputs.append(image)
                image = image[::2]


            result = layer(image)
            self.assertEqual(len(result),log_2)

            for ind,feature in enumerate(result):
                self.assertEqual(feature.shape[0],batch_size)
                self.assertEqual(feature.shape[1],out_channel)
                self.assertEqual(feature.shape[2],2**(log_2-ind))
                self.assertEqual(feature.shape[3],2**(log_2-ind))


    def test_forward_not_divisible_by_2(self):

        pass
