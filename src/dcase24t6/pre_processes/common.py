#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Any, Iterable

import torch
from torch import Tensor

from dcase24t6.utils.type_checks import is_list_int


def is_batch_audio(item_or_batch: dict[str, Any]) -> bool:
    audio = item_or_batch["audio"]
    sr = item_or_batch["sr"]

    if not isinstance(audio, Tensor):
        raise TypeError(
            f"Invalid audio input. (expected tensor but found {type(audio)})"
        )
    if isinstance(sr, Tensor):
        sr = sr.tolist()

    if audio.ndim == 2 and isinstance(sr, int):
        return False
    elif audio.ndim == 3 and is_list_int(sr):
        return True
    else:
        raise ValueError(
            "Invalid audio or sr input. (expected tensor with 3 dims and list of sampling rates or tensor with 2 dims and single sampling rate)"
        )


def batchify_audio(
    item_or_batch: dict[str, Any],
    input_time_dim: int = -1,
) -> tuple[dict[str, Any], bool, int]:
    was_batch = is_batch_audio(item_or_batch)

    if was_batch:
        # audio: (batch_size, channels, time_steps)
        batch = item_or_batch
        batch_size = batch["audio"].shape[0]
        return batch, was_batch, batch_size

    item_or_batch = copy.copy(item_or_batch)
    audio = item_or_batch.pop("audio")
    sr = item_or_batch.pop("sr")

    if isinstance(sr, Tensor):
        sr = sr.tolist()

    # audio: (channels, time_steps)
    batch_size = 1

    default_audio_lens = torch.full(
        (), audio.shape[input_time_dim], device=audio.device
    )
    audio_lens = item_or_batch.pop("audio_lens", default_audio_lens)

    audio = audio.unsqueeze(dim=0)
    audio_lens = audio_lens.unsqueeze(dim=0)

    sr = [sr]
    batch = item_or_batch | {"audio": audio, "audio_lens": audio_lens, "sr": sr}
    return batch, was_batch, batch_size


def unbatchify_audio(
    batch: dict[str, Any],
    was_batch: bool,
    keys: Iterable[str],
) -> dict[str, Any]:
    if was_batch:
        return batch

    batch = batch | {k: batch[k][0] for k in keys}
    return batch
