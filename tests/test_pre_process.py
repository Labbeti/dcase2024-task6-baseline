#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch
from torch import Tensor
from torchoutil import pad_and_stack_rec

from dcase24t6.pre_processes.cnext import ResampleMeanCNext
from dcase24t6.pre_processes.resample import Resample


class TestCNextPreProcess(TestCase):
    def test_resample_output_shapes(self) -> None:
        src_sr = 48_000
        device = "cpu"
        num_channels = 2
        gen = torch.Generator().manual_seed(42)

        bsize = 8
        durations = torch.empty(bsize).uniform_(1, 30, generator=gen).tolist()

        # durations = torch.arange(1, 11, 1).tolist()
        # bsize = len(durations)

        audio_lst = [
            torch.rand(
                num_channels, round(src_sr * duration), device=device, generator=gen
            )
            for duration in durations
        ]
        pre_process = Resample(32_000)

        frame_embs_per_item = []
        frame_embs_shape_per_item = []
        for audio_i in audio_lst:
            item_i = {"audio": audio_i, "sr": src_sr}
            output_i = pre_process(item_i)
            frame_embs_i = output_i["audio"]
            frame_embs_shape_i = output_i["audio_shape"]

            assert isinstance(frame_embs_i, Tensor)
            assert isinstance(frame_embs_shape_i, Tensor)
            assert torch.equal(
                torch.as_tensor(frame_embs_i.shape), frame_embs_shape_i
            ), f"{frame_embs_i.shape=}; {frame_embs_shape_i=}"

            frame_embs_per_item.append(frame_embs_i)
            frame_embs_shape_per_item.append(frame_embs_shape_i)

        audio_shape = [audio.shape for audio in audio_lst]
        audio_shape = torch.as_tensor(audio_shape, device=device)
        padded_audio = pad_and_stack_rec(audio_lst, pad_value=0.0)

        assert tuple(padded_audio.shape) == (
            bsize,
            num_channels,
            round(max(durations) * src_sr),
        )

        batch = {
            "audio": padded_audio,
            "audio_shape": audio_shape,
            "sr": [src_sr] * bsize,
        }
        output = pre_process(batch)

        frame_embs = output["audio"]
        frame_embs_shape = output["audio_shape"]
        assert isinstance(frame_embs, Tensor)
        assert isinstance(frame_embs_shape, Tensor)

        frame_embs_shape_per_item = torch.stack(frame_embs_shape_per_item)
        assert torch.allclose(
            frame_embs_shape, frame_embs_shape_per_item, atol=1
        ), f"Invalid shapes:\n{frame_embs_shape=} \n{frame_embs_shape_per_item=}"

    def test_cnext_output_shapes(self) -> None:
        src_sr = 48_000
        device = "cpu"
        num_channels = 2
        gen = torch.Generator().manual_seed(42)

        bsize = 8
        durations = torch.empty(bsize).uniform_(1, 30, generator=gen).tolist()

        # durations = torch.arange(1, 11, 1).tolist()
        # bsize = len(durations)

        audio_lst = [
            torch.rand(
                num_channels, round(src_sr * duration), device=device, generator=gen
            )
            for duration in durations
        ]
        pre_process = ResampleMeanCNext("cnext_bl", device=device)

        frame_embs_per_item = []
        frame_embs_shape_per_item = []
        for audio_i in audio_lst:
            item_i = {"audio": audio_i, "sr": src_sr}
            output_i = pre_process(item_i)
            frame_embs_i = output_i["frame_embs"]
            frame_embs_shape_i = output_i["frame_embs_shape"]

            assert isinstance(frame_embs_i, Tensor)
            assert isinstance(frame_embs_shape_i, Tensor)
            assert torch.equal(
                torch.as_tensor(frame_embs_i.shape), frame_embs_shape_i
            ), f"{frame_embs_i.shape=}; {frame_embs_shape_i=}"

            frame_embs_per_item.append(frame_embs_i)
            frame_embs_shape_per_item.append(frame_embs_shape_i)

        audio_shape = [audio.shape for audio in audio_lst]
        audio_shape = torch.as_tensor(audio_shape, device=device)
        padded_audio = pad_and_stack_rec(audio_lst, pad_value=0.0)

        assert tuple(padded_audio.shape) == (
            bsize,
            num_channels,
            round(max(durations) * src_sr),
        )

        batch = {
            "audio": padded_audio,
            "audio_shape": audio_shape,
            "sr": [src_sr] * bsize,
        }
        output = pre_process(batch)

        frame_embs = output["frame_embs"]
        frame_embs_shape = output["frame_embs_shape"]
        assert isinstance(frame_embs, Tensor)
        assert isinstance(frame_embs_shape, Tensor)

        frame_embs_shape_per_item = torch.stack(frame_embs_shape_per_item)
        assert torch.allclose(
            frame_embs_shape, frame_embs_shape_per_item, atol=1
        ), f"Invalid shapes:\n{frame_embs_shape=} \n{frame_embs_shape_per_item=}"


if __name__ == "__main__":
    unittest.main()
