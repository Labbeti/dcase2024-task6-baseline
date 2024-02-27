#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
from pathlib import Path
from typing import Callable, Iterable, Literal

import hydra
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.functional.clotho import download_clotho_datasets
from aac_metrics.download import download_metrics
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torchoutil.utils.hdf import pack_to_hdf

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="prepare",
)
def prepare(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    if cfg.verbose >= 1:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pre_transform = instantiate(cfg.pre_transform)

    return prepare_data_metrics_models(
        dataroot=cfg.path.data,
        subsets=cfg.subsets,
        force=cfg.force,
        hdf_pattern=cfg.hdf_pattern,
        pre_transform=pre_transform,
        overwrite=cfg.overwrite,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        verbose=cfg.verbose,
    )


def prepare_data_metrics_models(
    dataroot: str | Path = "data",
    subsets: Iterable[str] = (),
    force: bool = False,
    hdf_pattern: str = "{dataset}_{subset}.hdf",
    pre_transform: Callable | None = None,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int | Literal["auto"] = "auto",
    verbose: int = 0,
) -> None:
    dataroot = Path(dataroot).resolve()
    subsets = list(subsets)

    download_metrics(verbose=verbose)

    os.makedirs(dataroot, exist_ok=True)
    download_clotho_datasets(
        root=dataroot,
        subsets=subsets,
        force=force,
        verbose=verbose,
        clean_archives=False,
        verify_files=True,
    )

    hdf_root = dataroot.joinpath("HDF")
    os.makedirs(hdf_root, exist_ok=True)

    for subset in subsets:
        dataset = Clotho(
            root=dataroot,
            subset=subset,
            download=False,
            verbose=verbose,
        )

        # example: clotho_dev.hdf
        hdf_fname = hdf_pattern.format(
            dataset="clotho",
            subset=subset,
        )
        hdf_fpath = hdf_root.joinpath(hdf_fname)

        if hdf_fpath.exists() and not overwrite:
            continue

        pack_to_hdf(
            dataset=dataset,
            hdf_fpath=hdf_fpath,
            pre_transform=pre_transform,
            overwrite=overwrite,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )


if __name__ == "__main__":
    prepare()
