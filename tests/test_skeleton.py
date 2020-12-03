# -*- coding: utf-8 -*-

import pytest
from knn.skeleton import fib

__author__ = "Wojciech Przybyło, Artur Nowak, Monika Lis"
__copyright__ = " Monika Lis, Artur Nowak, Wojciech Przybyło"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
