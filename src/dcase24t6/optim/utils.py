#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable

from torch import nn


def create_params_groups(
    model: nn.Module,
    weight_decay: float,
    skip_list: Iterable[str] | None = (),
) -> list[dict[str, Any]]:
    if skip_list is None:
        skip_list = {}
    else:
        skip_list = dict.fromkeys(skip_list)

    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    params_groups = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    return params_groups
