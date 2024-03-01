#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Iterable, Mapping

import torch
from lightning import LightningDataModule, LightningModule, Trainer

from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

pylog = logging.getLogger(__name__)


class AACModel(LightningModule):
    def __init__(
        self,
        tokenizer: AACTokenizer | Iterable[AACTokenizer] | Mapping[str, AACTokenizer],
    ) -> None:
        if isinstance(tokenizer, AACTokenizer):
            tokenizers = {"0": tokenizer}
        elif isinstance(tokenizer, Mapping):
            tokenizers = dict(tokenizer.items())
        elif isinstance(tokenizer, Iterable):
            tokenizers = {
                f"{i}": tokenizer_i for i, tokenizer_i in enumerate(tokenizer)
            }
        else:
            raise TypeError(f"Invalid argument type {type(tokenizer)=}")

        super().__init__()
        self.tokenizers = tokenizers

    @property
    def tokenizer(self) -> AACTokenizer:
        if len(self.tokenizers) == 0:
            raise RuntimeError(
                "Cannot use property '.tokenizer' with a model that does not have any tokenizers."
            )
        if len(self.tokenizers) > 1:
            pylog.warning(
                f"You are using property '.tokenizer' but this model has more than 1 tokenizer. (found {len(self.tokenizers)} tokenizers)"
            )
        tokenizer = next(iter(self.tokenizers.values()))
        return tokenizer

    @property
    def dtype(self) -> torch.dtype:
        return super().dtype  # type: ignore

    def has_trainer(self) -> bool:
        return has_trainer(self)

    def has_datamodule(self) -> bool:
        return has_datamodule(self)

    @property
    def datamodule(self) -> LightningDataModule:
        if not self.has_datamodule():
            raise RuntimeError(
                "Cannot get property .datamodule because no datamodule is attached to current model."
            )
        return self.trainer.datamodule  # type: ignore


def has_trainer(plm: LightningModule) -> bool:
    return plm.trainer is not None


def has_datamodule(plm: LightningModule) -> bool:
    return has_trainer(plm) and trainer_has_datamodule(plm.trainer)


def trainer_has_datamodule(trainer: Trainer) -> bool:
    return trainer.datamodule is not None  # type: ignore
