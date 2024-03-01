#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "TRUE"

import logging
import os.path as osp
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import hydra
from hydra.utils import instantiate
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from dcase24t6.callbacks.evaluator import Evaluator
from dcase24t6.callbacks.opcounter import OpCounter
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer
from dcase24t6.utils.job import get_git_hash

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="train",
)
def train(cfg: DictConfig) -> None | float:
    seed_everything(cfg.seed)
    start_time = time.perf_counter()

    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    if cfg.verbose >= 1:
        logger.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize callbacks, model, etc...
    loggers: Logger | list[Logger] = instantiate(cfg.log)
    callbacks = get_callbacks(cfg)
    tokenizer: AACTokenizer = instantiate(cfg.tokenizer)
    datamodule: LightningDataModule = instantiate(cfg.datamodule, tokenizer=tokenizer)
    model: LightningModule = instantiate(cfg.model, tokenizer=tokenizer)
    trainer: Trainer = instantiate(
        cfg.trainer,
        callbacks=list(callbacks.values()),
        logger=loggers,
    )

    # Train
    trainer.validate(
        model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, verbose=cfg.verbose > 1
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, verbose=cfg.verbose > 1)
    trainer.predict(model, datamodule=datamodule)

    # Save job info & stats
    end_time = time.perf_counter()
    total_duration_s = end_time - start_time
    pretty_total_duration = str(timedelta(seconds=round(total_duration_s)))
    job_info = {
        "git_hash": get_git_hash(),
        "total_duration_s": total_duration_s,
        "total_duration": pretty_total_duration,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_stats(cfg.save_dir, tokenizer, datamodule, model, job_info)
    logger.info(
        f"Job results are saved in '{cfg.save_dir}'. (duration={pretty_total_duration})"
    )


def get_callbacks(cfg: DictConfig) -> dict[str, Callback]:
    checkpoint: ModelCheckpoint = instantiate(cfg.ckpt)
    # Avoid using '=' in filename because it mess up with hydra arguments
    checkpoint.CHECKPOINT_EQUALS_CHAR = "_"  # type: ignore

    evaluator: Evaluator = instantiate(cfg.evaluator)
    model_summary = ModelSummary(max_depth=1)
    op_counter = OpCounter(verbose=cfg.verbose)
    lr_monitor = LearningRateMonitor()

    callbacks: dict[str, Callback] = {
        "checkpoint": checkpoint,
        "evaluator": evaluator,
        "model_summary": model_summary,
        "op_counter": op_counter,
        "lr_monitor": lr_monitor,
    }

    if checkpoint.monitor is not None:
        early_stop = EarlyStopping(
            check_finite=True,
            mode=checkpoint.mode,
            monitor=checkpoint.monitor,
            patience=sys.maxsize,
        )
        callbacks["early_stop"] = early_stop

    callbacks_str = ", ".join(
        callback.__class__.__name__ for callback in callbacks.values()
    )
    logger.info(f"Adding {len(callbacks)} callbacks: {callbacks_str}")

    return callbacks


def save_stats(
    save_dir: str | Path,
    tokenizer: AACTokenizer,
    datamodule: LightningDataModule,
    model: LightningModule,
    job_info: Mapping[str, Any],
) -> None:
    save_dir = Path(save_dir).resolve()

    tok_fpath = save_dir.joinpath("tokenizer.json")
    tokenizer.save(tok_fpath)

    datamodule_fpath = save_dir.joinpath("hparams_datamodule.yaml")
    save_hparams_to_yaml(datamodule_fpath, dict(datamodule.hparams))

    model_fpath = save_dir.joinpath("hparams_model.yaml")
    save_hparams_to_yaml(model_fpath, dict(model.hparams))

    job_info_fpath = save_dir.joinpath("job_info.yaml")
    save_hparams_to_yaml(job_info_fpath, dict(job_info.items()))


if __name__ == "__main__":
    train()
