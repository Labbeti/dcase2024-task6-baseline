#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


def save_to_yaml(
    fpath: str | Path,
    data: Mapping[str, Any] | DictConfig,
    overwrite: bool = True,
    to_builtins: bool = True,
    resolve: bool = True,
    sort_keys: bool = False,
    **kwargs,
) -> None:
    fpath = Path(fpath).resolve()
    if not overwrite and fpath.exists():
        raise FileExistsError(f"File {fpath} already exists.")

    if resolve:
        if not isinstance(data, DictConfig):
            data = dict(data.items())
            data = OmegaConf.create(data)

        data = OmegaConf.to_container(data, resolve=True)  # type: ignore

    if to_builtins:
        data = {to_builtin(k): to_builtin(v) for k, v in data.items()}

    with open(fpath, "w", encoding="utf-8") as file:
        yaml.dump(data, file, sort_keys=sort_keys, **kwargs)


def save_to_csv(
    data: Iterable[Mapping[str, Any]],
    fpath: str | Path,
    overwrite: bool = True,
    to_builtins: bool = True,
    **kwargs,
) -> None:
    data = list(data)
    if len(data) <= 0:
        raise ValueError(f"Invalid argument {data=}. (found empty iterable)")

    fpath = Path(fpath).resolve()
    if not overwrite and fpath.exists():
        raise FileExistsError(f"File {fpath} already exists.")

    if to_builtins:
        data = [
            {to_builtin(k): to_builtin(v) for k, v in data_i.items()} for data_i in data
        ]

    fieldnames = list(data[0].keys())
    with open(fpath, "w") as file:
        writer = csv.DictWriter(file, fieldnames, **kwargs)
        writer.writeheader()
        writer.writerows(data)


def to_builtin(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    elif isinstance(v, tuple):
        return list(v)
    elif isinstance(v, Tensor):
        return v.tolist()
    else:
        return v
