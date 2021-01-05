from unittest import TestCase
from good_guys.models import LSTMCell
import torch


class TestLSTMCell(TestCase):
    def test_forward(self):

        cell = LSTMCell()
        batch_size = 1

        inp = torch.randn((batch_size,1024))
        prev_state = torch.randn(batch_size,128)

        s_t,p_t = cell(prev_state,inp)

        self.assertEqual(prev_state.shape,s_t.shape)
        self.assertEqual([1,1],list(p_t.shape))

        batch_size = 16
        inp = torch.randn((batch_size, 1024))
        prev_state = torch.randn(batch_size, 128)

        s_t, p_t = cell(prev_state, inp)

        self.assertEqual(prev_state.shape, s_t.shape)
        self.assertEqual([batch_size, 1], list(p_t.shape))
