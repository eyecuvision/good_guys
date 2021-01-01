from unittest import TestCase
from good_guys.layers import BiFPN
from itertools import product
import torch

class TestBiFPN(TestCase):
    FILTERS = [1,16,63]
    BATCH_SIZES = [1,16,63]
    LOG_2_VALUES = [4,5,6]

    def test_forward_divisible_by_2(self):

        # Bigger images have smaller indices


        for filters,batch_size,log_2 in product(self.FILTERS,self.BATCH_SIZES,self.LOG_2_VALUES):

            batch_size = 1
            layer = BiFPN(filters,log_2)
            image = torch.randn((batch_size,filters,2**log_2,2**log_2))

            inputs = []
            for i in range(log_2):
                inputs.append(image)
                image = image[:,:,::2,::2]


            result = layer(inputs)
            self.assertEqual(len(result),log_2)

            for ind,feature in enumerate(result):
                self.assertEqual(feature.shape[0],batch_size)
                self.assertEqual(feature.shape[1],filters)
                self.assertEqual(feature.shape[2],2**(log_2-ind))
                self.assertEqual(feature.shape[3],2**(log_2-ind))


    def test_forward_not_divisible_by_2(self):

        pass
