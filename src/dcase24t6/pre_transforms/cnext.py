#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchaudio.functional import resample
from torchoutil.nn.functional.get import get_device

from dcase24t6.nn.convnext import convnext_tiny
from dcase24t6.nn.convnext_ckpt_utils import load_cnext_state_dict


class ResampleMeanCNext(nn.Module):
    def __init__(
        self,
        model_name_or_path: str | Path,
        model_sr: int = 32_000,
        offline: bool = False,
        device: str | torch.device | None = "auto",
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
        audio = batch.pop("audio")
        audio_shape = batch.pop("audio_shape")
        sample_rates = batch["sr"]

        if not isinstance(audio, Tensor) or audio.ndim != 3:
            raise ValueError("Invalid audio input. (expected tensor with 3 dims)")
        if not all(sr == sample_rates[0] for sr in sample_rates[1:]):
            raise ValueError(
                f"Cannot transform a batch with audio sampled at different rates. (found {sample_rates=})"
            )

        audio = resample(audio, sample_rates[0], self.model_sr)

        # Remove channel dim
        audio = audio.mean(dim=1)
        audio = audio.to(device=self.device)

        model_outs = self.convnext(audio, audio_shape)
        frame_embs = model_outs["frame_embs"]
        return batch | {"audio": frame_embs}
