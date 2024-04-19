#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from dcase24t6.nn.hub import baseline_pipeline


class TestPipeline(TestCase):
    def test_example_1(self) -> None:
        model = baseline_pipeline(device="cpu")
        sr = 44100
        audio = torch.rand(1, 1, sr * 15)
        audio_shape = torch.as_tensor([audio[0].shape])
        batch = {"audio": audio, "audio_shape": audio_shape, "sr": [sr]}
        outputs = model(batch)

        assert isinstance(outputs, dict)
        print(outputs["candidates"])
        assert isinstance(outputs["candidates"], list)

    def test_example_2(self) -> None:
        model = baseline_pipeline(device="cpu")
        sr = 44100
        audio = torch.rand(1, sr * 15)
        item = {"audio": audio, "sr": sr}
        outputs = model(item)

        assert isinstance(outputs, dict)
        print(outputs["candidates"])
        assert isinstance(outputs["candidates"], list)


if __name__ == "__main__":
    unittest.main()
