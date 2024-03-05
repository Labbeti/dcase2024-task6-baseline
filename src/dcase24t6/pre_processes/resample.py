#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Any

from torch import nn
from torchaudio.functional import resample

from dcase24t6.pre_processes.common import batchify_audio, unbatchify_audio


class Resample(nn.Module):
    def __init__(
        self,
        target_sr: int = 32_000,
        input_time_dim: int = -1,
    ) -> None:
        super().__init__()
        self.target_sr = target_sr
        self.input_time_dim = input_time_dim

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch, was_batch, batch_size = batchify_audio(batch, self.input_time_dim)

        batch = copy.copy(batch)
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]
        sr = batch["sr"]

        if not all(sr_i == sr[0] for sr_i in sr[1:]):
            raise ValueError(
                f"Cannot transform a batch with audio sampled at different rates. (found {sr=})"
            )

        audio = resample(audio, sr[0], self.target_sr)

        resample_factor = self.target_sr / sr[0]
        audio_lens = (audio_lens * resample_factor).round()

        added = {
            "audio": audio,
            "audio_lens": audio_lens,
            "sr": [self.target_sr] * batch_size,
        }
        added = unbatchify_audio(added, was_batch, added.keys())
        outputs = batch | added
        return outputs
