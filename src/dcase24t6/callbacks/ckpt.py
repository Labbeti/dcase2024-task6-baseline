#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

from lightning import LightningModule, Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

pylog = logging.getLogger(__name__)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        # ModelCheckpoint args
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Literal[True, False, "link"]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        # Custom args
        replace_slash_in_filename: bool = True,
        create_best_symlink: bool = True,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        self.replace_slash_in_filename = replace_slash_in_filename
        self.create_best_symlink = create_best_symlink

    def _format_checkpoint_name(
        self,
        filename: Optional[str],
        metrics: dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        fname = super()._format_checkpoint_name(
            filename, metrics, prefix, auto_insert_metric_name
        )
        if self.replace_slash_in_filename:
            fname = fname.replace("/", "_")
        return fname

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_end(trainer, pl_module)

        if self.dirpath is not None:
            os.makedirs(self.dirpath, exist_ok=True)
            self.to_yaml()

        best_model_path = Path(self.best_model_path)
        if (
            not self.create_best_symlink
            or not trainer.is_global_zero
            or not best_model_path.is_file()
        ):
            return None

        ckpt_dpath = best_model_path.parent
        ckpt_fname = best_model_path.name
        lpath = ckpt_dpath.joinpath("best.ckpt")

        if lpath.exists():
            pylog.warning(f"Link {lpath.name} already exists.")
            return None

        os.symlink(ckpt_fname, lpath)

        if not lpath.is_file():
            pylog.error(f"Invalid symlink file {lpath=}.")
        elif self.verbose:
            pylog.debug(
                f"Create relative symlink for best model checkpoint '{lpath}'. (from='{self.best_model_path}')"
            )
