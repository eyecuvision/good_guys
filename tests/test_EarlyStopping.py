from unittest import TestCase
from good_guys.loss import EarlyStopping


class TestEarlyStopping(TestCase):

    def test_stop(self):

        number = 2
        stop = EarlyStopping(5,threshold= .5)

        for i in range(7):
            if stop(number):
                raise AssertionError("Stopped early.")
        for i in range(5):
            number += 1
            if stop(number):
                raise AssertionError("Stopped early.")
        else:
            if not stop(number):
                raise AssertionError("Didn't stop")
