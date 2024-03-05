#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torchaudio.functional import resample
from torchoutil.nn.functional.get import get_device

from dcase24t6.nn.encoders.convnext import convnext_tiny
from dcase24t6.nn.encoders.convnext_ckpt_utils import load_cnext_state_dict
from dcase24t6.utils.type_checks import is_list_int


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
        convnext.load_state_dict(state_dict, strict=True)

        for p in convnext.parameters():
            p.requires_grad_(False)
        convnext = convnext.eval()
        convnext = convnext.to(device=device)

        super().__init__()
        self.model_sr = model_sr
        self.convnext = convnext

    @property
    def device(self) -> torch.device:
        return self.convnext.device

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = copy.copy(batch)
        audio = batch.pop("audio")
        sr = batch["sr"]

        if not isinstance(audio, Tensor):
            raise TypeError("Invalid audio input. (expected tensor)")
        if isinstance(sr, Tensor):
            sr = sr.tolist()

        if audio.ndim == 2 and isinstance(sr, int):
            is_batch = False
            audio_shape = torch.as_tensor([audio.shape], device=self.device)
            audio = audio.unsqueeze(dim=0)
            sr = [sr]

        elif audio.ndim == 3 and (is_list_int(sr)):
            is_batch = True
            audio_shape = batch.pop("audio_shape")

        else:
            raise ValueError(
                "Invalid audio or sr input. (expected tensor with 3 dims and list of sampling rates or tensor with 2 dims and single sampling rate)"
            )

        if not all(sr_i == sr[0] for sr_i in sr[1:]):
            raise ValueError(
                f"Cannot transform a batch with audio sampled at different rates. (found {sr=})"
            )

        audio = resample(audio, sr[0], self.model_sr)

        # Remove channel dim
        audio = audio.mean(dim=1)
        audio = audio.to(device=self.device)

        model_outs = self.convnext(audio, audio_shape)
        if not is_batch:
            model_outs = {
                k: (v[0] if isinstance(v, Sequence) else v)
                for k, v in model_outs.items()
            }

        outputs = batch | model_outs
        return outputs
