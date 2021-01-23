from unittest import TestCase
from good_guys.loss import EarlyStopping


class TestEarlyStopping(TestCase):

    def test_stop(self):

        number = 2
        stop = EarlyStopping(5)

        for i in range(6):
            if stop(number):
                raise AssertionError("Stopped early.")
            number += 1
        else:
            if not stop(number):
                raise AssertionError("Didn't stop")
