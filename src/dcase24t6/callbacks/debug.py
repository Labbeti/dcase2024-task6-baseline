#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback

from dcase24t6.nn.functional import hash_model

pylog = logging.getLogger(__name__)


class PrintModelHash(Callback):
    """Print simple model hash for debugging purposes."""

    def __init__(
        self,
        save_dir: str | Path,
        hash_fname: str | Path = Path("outputs", "model_hash.txt"),
        verbose: int = 1,
    ) -> None:
        save_dir = Path(save_dir).resolve()

        super().__init__()
        self.save_dir = save_dir
        self.hash_fname = hash_fname
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_fit_start_{epoch}")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_train_start_{epoch}")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_validation_start_{epoch}")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_test_start_{epoch}")

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_predict_start_{epoch}")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_fit_end_{epoch}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_train_end_{epoch}")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_validation_end_{epoch}")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_test_end_{epoch}")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._hash_model(trainer, pl_module, "on_predict_end_{epoch}")

    def _hash_model(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step_name: str,
    ) -> int | None:
        if self.verbose < 2:
            return None

        step_name = step_name.format(epoch=trainer.current_epoch)

        training = pl_module.training
        pl_module.train(False)
        hash_value = hash_model(pl_module)
        pl_module.train(training)

        num_tensors = len(list(pl_module.parameters()))
        msg = f"Model hash at '{step_name}': {hash_value} ({num_tensors} tensors)"

        pylog.debug(msg)

        hash_fpath = self.save_dir.joinpath(self.hash_fname)
        with open(hash_fpath, "a") as file:
            file.writelines([msg + "\n"])

        return hash_value
