#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "FALSE"
os.environ["HF_HUB_OFFLINE"] = "FALSE"

import logging
import os
import os.path as osp
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import hydra
import nltk
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.functional.clotho import download_clotho_datasets
from aac_metrics.download import download_metrics
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data.dataset import Subset
from torchoutil.utils.hdf import HDFDataset, pack_to_hdf

from dcase24t6.callbacks.complexity import ComplexityProfiler
from dcase24t6.callbacks.emissions import CustomEmissionTracker
from dcase24t6.utils.job import get_git_hash
from dcase24t6.utils.saving import save_to_yaml

pylog = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="prepare",
)
def prepare(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    start_time = time.perf_counter()
    global_tracker: CustomEmissionTracker = instantiate(cfg.emission)
    global_tracker.start_task("total")

    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    if cfg.verbose >= 1:
        pylog.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    complexity_profiler = ComplexityProfiler(
        save_dir=cfg.save_dir,
        cplxity_fname="model_complexity_{dataset}_{subset}.yaml",
        verbose=cfg.verbose,
    )

    pre_process = instantiate(cfg.pre_process)

    hdf_datasets = prepare_data_metrics_models(
        dataroot=cfg.path.data_root,
        subsets=cfg.data.subsets,
        download_clotho=cfg.data.download,
        force=cfg.data.force,
        hdf_pattern=cfg.hdf_pattern,
        pre_process=pre_process,
        overwrite=cfg.overwrite,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        size_limit=cfg.size_limit,
        complexity_profiler=complexity_profiler,
        verbose=cfg.verbose,
    )

    # Save job info & stats
    global_tracker.stop_and_save_task("total")
    end_time = time.perf_counter()

    total_duration_s = end_time - start_time
    pretty_total_duration = str(timedelta(seconds=round(total_duration_s)))
    job_info = {
        "git_hash": get_git_hash(),
        "total_duration_s": total_duration_s,
        "total_duration": pretty_total_duration,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_prepare_stats(cfg.save_dir, hdf_datasets, job_info)
    pylog.info(
        f"Job results are saved in '{cfg.save_dir}'. (duration={pretty_total_duration})"
    )


def prepare_data_metrics_models(
    dataroot: str | Path = "data",
    subsets: Iterable[str] = (),
    download_clotho: bool = True,
    force: bool = False,
    hdf_pattern: str = "{dataset}_{subset}.hdf",
    pre_process: Callable | None = None,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int | Literal["auto"] = "auto",
    size_limit: int | None = None,
    complexity_profiler: ComplexityProfiler | None = None,
    verbose: int = 0,
) -> dict[str, HDFDataset]:
    dataroot = Path(dataroot).resolve()
    subsets = list(subsets)

    nltk.download("stopwords")
    download_metrics(verbose=verbose)

    os.makedirs(dataroot, exist_ok=True)

    if download_clotho:
        download_clotho_datasets(
            root=dataroot,
            subsets=subsets,
            force=force,
            verbose=verbose,
            clean_archives=False,
            verify_files=True,
        )

    hdf_datasets = {}

    for subset in subsets:
        dataset = Clotho(
            root=dataroot,
            subset=subset,
            download=False,
            verbose=verbose,
        )

        if size_limit is not None and len(dataset) > size_limit:
            dataset = Subset(dataset, list(range(size_limit)))

        # example: clotho_dev.hdf
        hdf_fname = hdf_pattern.format(
            dataset="clotho",
            subset=subset,
        )
        hdf_fpath = dataroot.joinpath("HDF", hdf_fname)
        os.makedirs(hdf_fpath.parent, exist_ok=True)

        if isinstance(pre_process, nn.Module) and complexity_profiler is not None:
            item = dataset[0]
            example = (item,)
            complexities = complexity_profiler.profile(example, pre_process)
            complexity_profiler.save(complexities, pre_process, item, fmt_kwargs=dict(dataset="clotho", subset=subset))  # type: ignore

        if hdf_fpath.exists() and not overwrite:
            pylog.info(
                f"Skipping {hdf_fname} because it already exists and {overwrite=}."
            )
            continue

        hdf_dataset = pack_to_hdf(
            dataset=dataset,  # type: ignore
            hdf_fpath=hdf_fpath,
            pre_transform=pre_process,
            overwrite=overwrite,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )
        hdf_datasets[hdf_fname] = hdf_dataset

    return hdf_datasets


def save_prepare_stats(
    save_dir: str | Path,
    hdf_datasets: dict[str, HDFDataset],
    job_info: Mapping[str, Any],
) -> None:
    save_dir = Path(save_dir).resolve()

    for hdf_fname, hdf_dataset in hdf_datasets.items():
        hdf_attrs_fname = f"hdf_attrs_{hdf_fname}.yaml".replace(".hdf", "")
        hdf_attrs_fpath = save_dir.joinpath(hdf_attrs_fname)
        save_to_yaml(hdf_dataset.attrs, hdf_attrs_fpath)

    job_info_fpath = save_dir.joinpath("job_info.yaml")
    save_to_yaml(job_info, job_info_fpath, resolve=True)


if __name__ == "__main__":
    prepare()
