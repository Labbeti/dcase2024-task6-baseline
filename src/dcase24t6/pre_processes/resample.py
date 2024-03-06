#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Any

from torch import Tensor, nn
from torchaudio.functional import resample

from dcase24t6.pre_processes.common import batchify, is_audio_batch, unbatchify


class Resample(nn.Module):
    def __init__(
        self,
        target_sr: int = 32_000,
        input_time_dim: int = -1,
    ) -> None:
        super().__init__()
        self.target_sr = target_sr
        self.input_time_dim = input_time_dim

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
        batch = copy.copy(batch)
        audio: Tensor = batch["audio"]
        audio_shape: Tensor = batch["audio_shape"]
        sr = batch["sr"]
        if isinstance(sr, Tensor):
            sr = sr.tolist()
        sr: list[int]

        if not all(sr_i == sr[0] for sr_i in sr[1:]):
            raise ValueError(
                f"Cannot transform a batch with audio sampled at different rates. (found {sr=})"
            )

        src_maxlen = audio.shape[self.input_time_dim]
        audio = resample(audio, sr[0], self.target_sr)
        tgt_maxlen = audio.shape[self.input_time_dim]

        reduction_factor = tgt_maxlen / src_maxlen

        audio_lens = audio_shape[:, self.input_time_dim]
        audio_lens = (audio_lens * reduction_factor).round().int()
        audio_shape[:, self.input_time_dim] = audio_lens

        batch_size = audio.shape[0]
        outputs = {
            "audio": audio,
            "audio_shape": audio_shape,
            "sr": [self.target_sr] * batch_size,
        }
        return batch | outputs
