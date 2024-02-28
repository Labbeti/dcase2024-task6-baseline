#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

import hydra
from hydra.utils import instantiate
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from dcase24t6.callbacks.opcounter import OpCounter

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="train",
)
def train(cfg: DictConfig) -> None | float:
    seed_everything(cfg.seed)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    if cfg.verbose >= 1:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    loggers: Logger | list[Logger] = instantiate(cfg.log)
    callbacks = get_callbacks(cfg)

    tokenizer = instantiate(cfg.tokenizer)
    datamodule = instantiate(cfg.datamodule, tokenizer=tokenizer)
    model = instantiate(cfg.model, tokenizer=tokenizer)
    trainer: Trainer = instantiate(
        cfg.trainer,
        callbacks=list(callbacks.values()),
        logger=loggers,
    )

    trainer.validate(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def get_callbacks(cfg: DictConfig) -> dict[str, Callback]:
    checkpoint = instantiate(cfg.ckpt)
    # evaluator = Evaluator(cfg.logdir)
    model_summary = ModelSummary(max_depth=1)
    opcounter = OpCounter(verbose=cfg.verbose)

    callbacks: dict[str, Callback] = {
        "checkpoint": checkpoint,
        # "evaluator": evaluator,
        "model_summary": model_summary,
        "opcounter": opcounter,
    }
    return callbacks


if __name__ == "__main__":
    train()
