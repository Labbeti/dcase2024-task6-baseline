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
def train(cfg: DictConfig) -> None | float:
    callbacks = get_callbacks(cfg)

    tokenizer = instantiate(cfg.tokenizer)
    datamodule = instantiate(cfg.datamodule, tokenizer=tokenizer)
    model = instantiate(cfg.model, tokenizer=tokenizer)
    trainer: Trainer = instantiate(cfg.trainer, callbacks=list(callbacks.values()))

    trainer.validate(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def get_callbacks(cfg: DictConfig) -> dict[str, Callback]:
    checkpoint = instantiate(cfg.ckpt)

    evaluator = Evaluator(cfg.logdir)
    callbacks: dict[str, Callback] = {
        "evaluator": evaluator,
        "checkpoint": checkpoint,
    }

    return callbacks


if __name__ == "__main__":
    train()
