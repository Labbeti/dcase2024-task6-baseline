#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Literal

from deepspeed.profiling.flops_profiler import get_model_profile
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from torchoutil import move_to_rec

from dcase24t6.utils.saving import save_to_yaml

pylog = logging.getLogger(__name__)


class OpCounter(Callback):
    def __init__(
        self,
        save_dir: str | Path,
        cplxity_fname: str = "model_complexity.yaml",
        backend: Literal["deepspeed"] = "deepspeed",
        verbose: int = 1,
    ) -> None:
        save_dir = Path(save_dir).resolve()
        super().__init__()
        self.save_dir = save_dir
        self.cplxity_fname = cplxity_fname
        self.backend = backend
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        datamodule = trainer.datamodule  # type: ignore
        if "batch_size" not in datamodule.hparams:
            pylog.warning(
                "Cannot compute FLOPs or MACs since datamodule does not have batch_size hyperparameter."
            )
            return None

        batch_size = datamodule.hparams["batch_size"]
        datamodule.hparams["batch_size"] = 1
        loader = datamodule.train_dataloader()
        datamodule.hparams["batch_size"] = batch_size

        example = {"batch": next(iter(loader))}
        self.count_num_ops(pl_module, example, "train")

    def count_num_ops(
        self,
        pl_module: LightningModule,
        example: Any,
        stage: str,
    ) -> None:
        match self.backend:
            case "deepspeed":
                flops, macs, params = measure_complexity_with_deepspeed(
                    pl_module, example, self.verbose
                )
            case backend:
                raise NotImplementedError(f"Invalid argument {backend}.")

        metrics = {
            "model/flops": flops,
            "model/macs": macs,
            "model/params": params,
        }
        for pl_logger in pl_module.loggers:
            pl_logger.log_metrics(metrics)  # type: ignore

        cplxity_fname = self.cplxity_fname.format(stage=stage)
        cplxity_fpath = self.save_dir.joinpath(cplxity_fname)
        save_to_yaml(metrics, cplxity_fpath)


def measure_complexity_with_deepspeed(
    model: LightningModule,
    example: Any,
    verbose: int = 0,
) -> tuple[int, int, int]:
    example = move_to_rec(example, device=model.device)
    flops, macs, params = get_model_profile(
        model,
        kwargs=example,
        print_profile=verbose >= 2,
        as_string=False,
    )
    return flops, macs, params  # type: ignore
