#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hydra
from hydra.utils import instantiate
from lightning import Callback, Trainer
from omegaconf import DictConfig

from dcase24t6.callbacks.evaluator import Evaluator


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="train",
)
def fit(cfg: DictConfig) -> None:
    callbacks = get_callbacks()

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer, callbacks=list(callbacks.values()))

    trainer.validate(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def get_callbacks() -> dict[str, Callback]:
    evaluator = Evaluator()
    callbacks: dict[str, Callback] = {
        "evaluator": evaluator,
    }
    return callbacks


if __name__ == "__main__":
    fit()
