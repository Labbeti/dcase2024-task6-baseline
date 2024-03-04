#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
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
        emissions_fname: str = "{task}_emissions.yaml",
        offline: bool = False,
        country_iso_code: str | None = None,
        disabled: bool = False,
        **kwargs,
    ) -> None:
        save_dir = Path(save_dir).resolve()

        kwds: dict[str, Any] = dict(
            project_name=dcase24t6.__name__,
            log_level=logging.WARNING,
            save_to_file=True,
            **kwargs,
        )
        if disabled:
            tracker = None
        elif offline:
            if country_iso_code is None:
                raise ValueError(
                    f"Invalid argument {country_iso_code=} with {offline=}. You must provide a Country code or use internet connection."
                )

            kwds["country_iso_code"] = country_iso_code
            tracker = OfflineEmissionsTracker(**kwds)
        else:
            tracker = EmissionsTracker(**kwds)

        super().__init__()
        self.save_dir = save_dir
        self.emissions_fname = emissions_fname
        self.tracker = tracker

    def is_disabled(self) -> bool:
        return self.tracker is None

    def start_task(self, task: str) -> None:
        if self.tracker is None:
            return None
        else:
            return self.tracker.start_task(task)

    def stop_task(self, task: str) -> EmissionsData | None:
        if self.tracker is None:
            return None
        else:
            return self.tracker.stop_task(task)

    def stop_and_save_task(self, task: str) -> EmissionsData | None:
        emissions = self.stop_task(task)
        if emissions is None:
            return emissions
        self.save_task(task, emissions)

    def save_task(self, task: str, emissions: EmissionsData) -> None:
        emissions_fname = self.emissions_fname.format(task=task)
        emissions_fpath = self.save_dir.joinpath(emissions_fname)
        save_to_yaml(emissions, emissions_fpath, resolve=False, make_parents=True)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_task("fit")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_task("test")

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_task("predict")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.stop_and_save_task("fit")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.stop_and_save_task("test")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.stop_and_save_task("predict")
