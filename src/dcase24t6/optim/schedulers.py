#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

pylog = logging.getLogger(__name__)


class CosDecayScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        n_steps: int,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_lambda=CosDecayRule(n_steps),
            last_epoch=last_epoch,
        )


class CosDecayRule:
    # Note : use class instead of function for scheduler rules to become pickable for multiple-GPU with Lightning
    def __init__(self, n_steps: int) -> None:
        if n_steps < 0:
            raise ValueError(
                f"Invalid argument {n_steps=} < 0 in {self.__class__.__name__}."
            )
        elif n_steps == 0:
            pylog.warning(
                f"Replacing {n_steps=} by n_steps=1 in {self.__class__.__name__}."
            )
            n_steps = max(n_steps, 1)
        super().__init__()
        self.n_steps = n_steps

    def __call__(self, step: int) -> float:
        step = min(step, self.n_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * step / self.n_steps))
