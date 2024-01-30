#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from lightning import LightningModule


class AACModel(LightningModule):
    @property
    def dtype(self) -> torch.dtype:
        return super().dtype  # type: ignore
