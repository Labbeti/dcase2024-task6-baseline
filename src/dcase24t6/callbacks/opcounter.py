#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from lightning import LightningModule
from lightning.pytorch.callbacks.callback import Callback

logger = logging.getLogger(__name__)


class OpCounter(Callback):
    def __init__(self, verbose: int = 1) -> None:
        super().__init__()
        self.verbose = verbose

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        self.count_num_ops(pl_module)

    def count_num_ops(self, pl_module: LightningModule) -> None:
        example = pl_module.example_input_array
        if example is None:
            logger.warning(
                "Cannot compute FLOPs or MACs: no example is attached to model."
            )

        # TODO
