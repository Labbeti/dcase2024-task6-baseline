#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import yaml
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from dcase24t6.utils.collections import dict_list_to_list_dict


@runtime_checkable
class DataclassInstance(Protocol):
    # Class meant for typing purpose only
    __dataclass_fields__: ClassVar[dict[str, Any]]


@runtime_checkable
class NamedTupleInstance(Protocol):
    # Class meant for typing purpose only
    _fields: tuple[str, ...]
    _fields_defaults: dict[str, Any]

    def _asdict(self) -> dict[str, Any]:
        ...


def save_to_yaml(
    data: (
        Mapping[str, Any]
        | DictConfig
        | Namespace
        | DataclassInstance
        | NamedTupleInstance
    ),
    fpath: str | Path | None,
    *,
    overwrite: bool = True,
    to_builtins: bool = True,
    resolve: bool = True,
    sort_keys: bool = False,
    indent: int | None = 4,
    make_parents: bool = True,
    **kwargs,
) -> str:
    if fpath is not None:
        fpath = Path(fpath).resolve()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            os.makedirs(fpath.parent, exist_ok=True)

    if isinstance(data, Namespace):
        data = data.__dict__

    elif is_dataclass(data) or isinstance(data, DataclassInstance):
        if isinstance(data, type):
            raise TypeError(f"Invalid argument type {type(data)}.")
        data = asdict(data)

    elif isinstance(data, NamedTupleInstance):
        data = data._asdict()

    if resolve:
        if isinstance(data, DictConfig):
            data_cfg = data
        else:
            data = dict(data.items())
            data_cfg = OmegaConf.create(data)

        data = OmegaConf.to_container(data_cfg, resolve=True)  # type: ignore

    elif isinstance(data, DictConfig):
        data = OmegaConf.to_container(data_cfg)  # type: ignore

    if to_builtins:
        data = {to_builtin(k): to_builtin(v) for k, v in data.items()}  # type: ignore

    content = yaml.dump(data, sort_keys=sort_keys, indent=indent, **kwargs)
    if fpath is not None:
        fpath.write_text(content, encoding="utf-8")
    return content


def save_to_csv(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    fpath: str | Path,
    *,
    overwrite: bool = True,
    to_builtins: bool = True,
    make_parents: bool = True,
    **kwargs,
) -> None:
    if isinstance(data, Mapping):
        data = dict_list_to_list_dict(data)
    else:
        data = list(data)

    if len(data) <= 0:
        raise ValueError(f"Invalid argument {data=}. (found empty iterable)")

    fpath = Path(fpath).resolve()
    if not overwrite and fpath.exists():
        raise FileExistsError(f"File {fpath} already exists.")
    elif make_parents:
        os.makedirs(fpath.parent, exist_ok=True)

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
