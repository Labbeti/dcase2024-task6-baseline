#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchoutil.nn.functional.get import get_device

from dcase24t6.nn.encoders.convnext import convnext_tiny
from dcase24t6.nn.encoders.convnext_ckpt_utils import load_cnext_state_dict
from dcase24t6.pre_processes.common import batchify_audio, unbatchify_audio
from dcase24t6.pre_processes.resample import Resample


class ResampleMeanCNext(nn.Module):
    """Offline transform applied to audio inputs for trans_decoder model.

    This modules handle single example and batch of examples as input.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
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
        state_dict = load_cnext_state_dict(
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

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = self.resample(batch)

        batch, was_batch, _batch_size = batchify_audio(batch, self.input_time_dim)

        batch = copy.copy(batch)
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]

        # Remove channel dim
        audio = audio.mean(dim=1)
        audio = audio.to(device=self.device)

        # audio: (bsize, reduced_time_steps)
        # audio_lens: (bsize,)

        added = self.convnext(audio, audio_lens)

        added = unbatchify_audio(
            added, was_batch, keys=["frame_embs", "frame_embs_lens", "sr"]
        )
        outputs = batch | added
        return outputs
