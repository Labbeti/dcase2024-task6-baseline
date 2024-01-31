#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
from lightning import LightningModule

from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

logger = logging.getLogger(__name__)


class AACModel(LightningModule):
    def __init__(self, tokenizer: AACTokenizer) -> None:
        super().__init__()
        self.tokenizers = {"0": tokenizer}

    @property
    def tokenizer(self) -> AACTokenizer:
        if len(self.tokenizers) == 0:
            raise RuntimeError(
                "Cannot use property '.tokenizer' with a model that does not have any tokenizers."
            )
        if len(self.tokenizers) > 1:
            logger.warning(
                f"You are using property '.tokenizer' but this model has more than 1 tokenizer. (found {len(self.tokenizers)} tokenizers)"
            )
        tokenizer = next(iter(self.tokenizers.values()))
        return tokenizer

    @property
    def dtype(self) -> torch.dtype:
        return super().dtype  # type: ignore
