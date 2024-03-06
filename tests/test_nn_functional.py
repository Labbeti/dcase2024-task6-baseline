#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import unittest
from unittest import TestCase

import torch
from torch import Size

from dcase24t6.nn.functional import remove_index_nd


class TestFunctional(TestCase):
    def test_remove_index_nd(self) -> None:
        for _ in range(100):
            ndim = int(torch.randint(1, 4, ()).item())
            in_shape = torch.randint(1, 10, (ndim,)).tolist()
            dim = int(torch.randint(0, len(in_shape), ()).item())
            index = int(torch.randint(0, in_shape[dim], ()).item())

            expected_shape = copy.copy(in_shape)
            expected_shape[dim] -= 1
            expected_shape = Size(expected_shape)

            t1 = torch.rand(in_shape)
            t2 = remove_index_nd(t1, index, dim=dim)

            assert expected_shape == t2.shape


if __name__ == "__main__":
    unittest.main()
