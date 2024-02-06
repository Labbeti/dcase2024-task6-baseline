#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Callable, Iterable, Literal

import hydra
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.functional.clotho import download_clotho_datasets
from aac_metrics.download import download_metrics
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchoutil.utils.hdf import pack_to_hdf


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="prepare",
)
def prepare(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    pre_transform_name = hydra_cfg.runtime.choices["pre_transform"]
    pre_transform = instantiate(cfg.pre_transform)
    hdf_pattern = f"{{dataset}}_{{subset}}_{pre_transform_name}.hdf"

    return prepare_data_metrics_models(
        dataroot=cfg.dataroot,
        subsets=cfg.subsets,
        force=cfg.force,
        hdf_pattern=hdf_pattern,
        pre_transform=pre_transform,
        verbose=cfg.verbose,
    )


def prepare_data_metrics_models(
    dataroot: str | Path,
    subsets: Iterable[str] = (),
    force: bool = False,
    hdf_pattern: str = "{dataset}_{subset}.hdf",
    pre_transform: Callable | None = None,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int | Literal["auto"] = "auto",
    verbose: int = 0,
) -> None:
    dataroot = Path(dataroot)
    subsets = list(subsets)

    download_metrics(verbose=verbose)
    download_clotho_datasets(
        root=dataroot,
        subsets=subsets,
        force=force,
        verbose=verbose,
        clean_archives=True,
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

        # example: clotho_dev_cnext.hdf
        hdf_fname = hdf_pattern.format(
            dataset="clotho",
            subset=subset,
        )

        hdf_fpath = hdf_root.joinpath(hdf_fname)
        if hdf_fpath.exists() and not overwrite:
            continue

        pack_to_hdf(
            dataset,
            hdf_fpath,
            pre_transform,
            overwrite=overwrite,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )


if __name__ == "__main__":
    prepare()
