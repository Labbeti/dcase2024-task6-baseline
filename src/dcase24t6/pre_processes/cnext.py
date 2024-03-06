#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchoutil.nn.functional.get import get_device

from dcase24t6.nn.ckpt import CNextRegister
from dcase24t6.nn.encoders.convnext import convnext_tiny
from dcase24t6.nn.functional import remove_index_nd
from dcase24t6.pre_processes.common import batchify, is_audio_batch, unbatchify
from dcase24t6.pre_processes.resample import Resample


class ResampleMeanCNext(nn.Module):
    """Offline transform applied to audio inputs for trans_decoder model.

    This modules handle single example and batch of examples as input.
    """

    def __init__(
        self,
        model_name_or_path: str | Path = "cnext_bl",
        model_sr: int = 32_000,
        offline: bool = False,
        device: str | torch.device | None = "cuda_if_available",
        input_time_dim: int = -1,
    ) -> None:
        device = get_device(device)

        convnext = convnext_tiny(
            pretrained=False,
            strict=False,
            drop_path_rate=0.0,
            after_stem_dim=[252, 56],
            use_speed_perturb=False,
            waveform_input=True,
            return_frame_outputs=True,
            return_clip_outputs=True,
        )
        state_dict = CNextRegister.load_state_dict(
            model_name_or_path,
            device="cpu",
            offline=offline,
        )
        convnext.load_state_dict(state_dict)

        for p in convnext.parameters():
            p.requires_grad_(False)
        convnext = convnext.eval()
        convnext = convnext.to(device=device)

        super().__init__()
        self.convnext = convnext
        self.input_time_dim = input_time_dim
        self.resample = Resample(model_sr, input_time_dim)

    @property
    def device(self) -> torch.device:
        return self.convnext.device

    def forward(self, item_or_batch: dict[str, Any]) -> dict[str, Any]:
        if is_audio_batch(item_or_batch):
            return self.forward_batch(item_or_batch)
        else:
            item = item_or_batch
            batch = batchify(item)
            batch = self.forward_batch(batch)
            item = unbatchify(batch)
            return item

    def forward_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = self.resample(batch)

        batch = copy.copy(batch)
        audio = batch.pop("audio")
        audio_shape = batch.pop("audio_shape")

        # audio: (bsize, channels, reduced_time_steps)
        # audio_shape: (bsize, 2)

        # Remove channel dim
        channel_dim = 1
        audio = audio.mean(dim=channel_dim)
        audio_shape = remove_index_nd(audio_shape, index=channel_dim - 1, dim=1)

        # audio: (bsize, reduced_time_steps)
        # audio_shape: (bsize, 1)

        audio = audio.to(device=self.device)
        outputs = self.convnext(audio, audio_shape)
        return batch | outputs
