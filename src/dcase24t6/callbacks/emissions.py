#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Any, Literal

from codecarbon.emissions_tracker import (
    EmissionsData,
    EmissionsTracker,
    OfflineEmissionsTracker,
)
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

import dcase24t6
from dcase24t6.utils.saving import save_to_yaml

EmissionStage = Literal["fit", "test", "predict"]


class CustomEmissionTracker(Callback):
    def __init__(
        self,
        save_dir: str | Path,
        experiment_name: str,
        offline: bool = False,
        country_iso_code: str | None = None,
        **kwargs,
    ) -> None:
        save_dir = Path(save_dir).resolve()
        output_dir = save_dir.joinpath("emissions")

        kwds: dict[str, Any] = dict(
            project_name=dcase24t6.__name__,
            output_dir=str(output_dir),
            experiment_name=experiment_name,
            log_level=logging.WARNING,
            save_to_file=True,
            **kwargs,
        )
        if offline:
            if country_iso_code is None:
                raise ValueError(
                    f"Invalid argument {country_iso_code=} with {offline=}. You must provide a Country code or use internet connection."
                )

            kwds["country_iso_code"] = country_iso_code
            tracker = OfflineEmissionsTracker(**kwds)
        else:
            tracker = EmissionsTracker(**kwds)

        super().__init__()
        self.output_dir = output_dir
        self.tracker = tracker

    def start_task(self, task: str) -> None:
        return self.tracker.start_task(task)

    def stop_task(self, task: str) -> EmissionsData:
        return self.tracker.stop_task(task)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start("fit")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start("test")

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start("predict")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_end("fit")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_end("test")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_end("predict")

    def _on_start(self, stage: EmissionStage) -> None:
        self.start_task(stage)

    def _on_end(self, stage: EmissionStage) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        emissions = self.stop_task(stage)
        emissions_fname = "{stage}_emissions.yaml".format(stage=stage)
        emissions_fpath = self.output_dir.joinpath(emissions_fname)
        save_to_yaml(emissions, emissions_fpath, resolve=False)
