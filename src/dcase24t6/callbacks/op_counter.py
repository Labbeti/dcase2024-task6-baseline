#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from torch import nn
from torchoutil import get_device, move_to_rec

from dcase24t6.models.aac import TestBatch
from dcase24t6.utils.saving import save_to_yaml
from dcase24t6.utils.type_checks import is_list_str

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

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        datamodule = trainer.datamodule  # type: ignore
        if "batch_size" not in datamodule.hparams:
            pylog.warning(
                "Cannot compute FLOPs or MACs since datamodule does not have batch_size hyperparameter."
            )
            return None

        source_batch_size = datamodule.hparams["batch_size"]
        target_batch_size = 1
        datamodule.hparams["batch_size"] = target_batch_size
        loaders = datamodule.test_dataloader()
        datamodule.hparams["batch_size"] = source_batch_size

        dataloader_idx = 0
        if isinstance(loaders, list):
            loader = loaders[dataloader_idx]
        else:
            loader = loaders
        del loaders
        batch: TestBatch = next(iter(loader))

        if "mult_captions" in batch and "captions" not in batch:
            batch["captions"] = batch["mult_captions"][:, 0]  # type: ignore

        METHODS = ("forcing", "generate")
        complexities = {}
        for method in METHODS:
            example = {
                "batch": batch,
                "method": method,
            }
            example_complexities = self.profile(example, pl_module, pl_module.device)
            complexities |= {
                f"{method}_{k}": v for k, v in example_complexities.items()
            }
        self.save(complexities, pl_module, batch)

    def profile(
        self,
        example: Any,
        model: nn.Module,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        if device is None and hasattr(model, "device"):
            device = model.device
        else:
            model = model.to(device=device)

        complexities = {}

        start = time.perf_counter()
        match self.backend:
            case "deepspeed":
                flops, macs, params = _measure_complexity_with_deepspeed(
                    model=model,
                    example=example,
                    device=device,
                    verbose=self.verbose,
                )
            case backend:
                raise ValueError(f"Invalid argument {backend=}.")
        end = time.perf_counter()

        complexities = {
            "params": params,
            "flops": flops,
            "macs": macs,
            "duration": end - start,
        }
        return complexities

    def save(
        self,
        complexities: dict[str, Any],
        model: nn.Module,
        batch: TestBatch,
        fmt_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(model, LightningModule):
            for pl_logger in model.loggers:
                pl_logger.log_metrics(complexities)  # type: ignore

        batch_size = len(batch["fname"]) if is_list_str(batch["fname"]) else 1
        cplxity_info = {
            "metrics": complexities,
            "model_class": model.__class__.__name__,
            "batch_size": batch_size,
            "fname": batch["fname"],
            "dataset": batch["dataset"],
            "subset": batch["subset"],
        }

        if fmt_kwargs is None:
            fmt_kwargs = {}
        cplxity_fname = self.cplxity_fname.format(**fmt_kwargs)
        cplxity_fpath = self.save_dir.joinpath(cplxity_fname)
        save_to_yaml(cplxity_info, cplxity_fpath)


def _measure_complexity_with_deepspeed(
    model: nn.Module,
    example: Any,
    device: str | torch.device | None,
    verbose: int = 0,
) -> tuple[int, int, int]:
    device = get_device(device)
    example = move_to_rec(example, device=device)
    flops, macs, params = get_model_profile(
        model,
        kwargs=example,
        print_profile=verbose >= 2,
        detailed=True,
        as_string=False,
    )
    return flops, macs, params  # type: ignore
