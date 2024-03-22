#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, TypedDict

import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from torch import nn
from torchoutil import get_device, move_to_rec

from dcase24t6.models.aac import TestBatch
from dcase24t6.utils.saving import save_to_yaml
from dcase24t6.utils.type_checks import is_list_str

pylog = logging.getLogger(__name__)


class ProfileOutput(TypedDict):
    model_output: Any
    params: int
    macs: int
    flops: int
    duration: float


class ComplexityProfiler(Callback):
    def __init__(
        self,
        save_dir: str | Path,
        cplxity_fname: str | Path = "model_complexity.yaml",
        profile_fname: str | Path = Path("outputs", "profile.txt"),
        backend: Literal["deepspeed"] = "deepspeed",
        verbose: int = 1,
    ) -> None:
        save_dir = Path(save_dir).resolve()
        super().__init__()
        self.save_dir = save_dir
        self.cplxity_fname = cplxity_fname
        self.profile_fname = profile_fname
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

        batch = move_to_rec(batch, device=pl_module.device)  # type: ignore

        METHODS = ("forcing", "generate")
        complexities = {}
        for method in METHODS:
            example = {
                "batch": batch,
                "method": method,
            }
            example_complexities = self.profile(example, pl_module, pl_module.device)
            example_complexities.pop("model_output")  # type: ignore

            complexities |= {
                f"{method}_{k}": v for k, v in example_complexities.items()
            }

        self.save(complexities, pl_module, batch)

    def profile(
        self,
        example: Mapping[str, Any] | tuple,
        model: nn.Module,
        device: str | torch.device | None = None,
    ) -> ProfileOutput:
        match self.backend:
            case "deepspeed":
                profile_fname = str(self.profile_fname)
                profile_fpath = self.save_dir.joinpath(profile_fname)

                output = _measure_complexity_with_deepspeed(
                    model=model,
                    example=example,
                    device=device,
                    verbose=self.verbose,
                    output_file=profile_fpath,
                )

            case backend:
                raise ValueError(f"Invalid argument {backend=}.")

        return output

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

        cplxity_fname = str(self.cplxity_fname)
        cplxity_fname = cplxity_fname.format(**fmt_kwargs)
        cplxity_fpath = self.save_dir.joinpath(cplxity_fname)
        save_to_yaml(cplxity_info, cplxity_fpath)


def _measure_complexity_with_deepspeed(
    model: nn.Module,
    example: Mapping[str, Any] | tuple,
    device: str | torch.device | None,
    module_depth: int = -1,
    top_modules: int = 1,
    warm_up: int = 1,
    output_file: str | Path | None = None,
    ignore_modules: Iterable[type] | None = None,
    verbose: int = 0,
) -> ProfileOutput:
    if device is None and hasattr(model, "device"):
        device = model.device
        device = get_device(device)
    else:
        device = get_device(device)
        model = model.to(device=device)

    example = move_to_rec(example, device=device)

    if isinstance(example, Mapping):
        args = []
        kwargs = example
    elif isinstance(example, tuple):
        args = example
        kwargs = {}
    else:
        raise TypeError(
            f"Invalid argument type {type(example)=}. (expected mapping or tuple)"
        )

    prof = FlopsProfiler(model)
    model.eval()

    if verbose >= 2:
        pylog.info("Flops profiler warming-up...")

    for _ in range(warm_up):
        _ = model(*args, **kwargs)

    prof.start_profile(ignore_list=ignore_modules)

    output = model(*args, **kwargs)

    flops: int = prof.get_total_flops()  # type: ignore
    macs: int = prof.get_total_macs()  # type: ignore
    params: int = prof.get_total_params()  # type: ignore
    duration: float = prof.get_total_duration()  # type: ignore

    if verbose >= 2 or output_file is not None:
        prof.print_model_profile(
            profile_step=warm_up,
            module_depth=module_depth,
            top_modules=top_modules,
            detailed=verbose >= 2,
            output_file=output_file,
        )
    prof.end_profile()

    return {
        "model_output": output,
        "flops": flops,
        "macs": macs,
        "params": params,
        "duration": duration,
    }
