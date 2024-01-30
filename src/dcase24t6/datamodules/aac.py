#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lightning import LightningDataModule


class AACDatamodule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
