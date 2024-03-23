#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Any, Sequence, TypedDict

import torch
from torch import Tensor

from dcase24t6.utils.type_checks import is_list_int


class BatchLike(TypedDict):
    audio: Tensor
    sr: list[int] | Tensor


def is_audio_batch(item_or_batch: dict[str, Any]) -> bool:
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


def batchify(item: dict[str, Any]) -> dict[str, list | Tensor]:
    """Transform a item dict to a batch dict."""
    item = add_audio_shape_to_item(item)
    result = {}
    for k, v in item.items():
        if isinstance(v, Tensor):
            result[k] = v.unsqueeze(dim=0)
        else:
            result[k] = [v]
    return result


def unbatchify(batch: dict[str, list | Tensor]) -> dict[str, Any]:
    result = {}
    for k, v in batch.items():
        if isinstance(v, (Sequence, Tensor)):
            result[k] = v[0]
        else:
            result[k] = v
    return result


def add_audio_shape_to_item(item: dict[str, Any]) -> dict[str, Any]:
    if "audio_shape" in item:
        return item

    audio = item["audio"]
    item = copy.copy(item)
    item["audio_shape"] = torch.as_tensor(audio.shape, device=audio.device)
    return item
