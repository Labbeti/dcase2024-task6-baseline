#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "FALSE"
os.environ["HF_HUB_OFFLINE"] = "FALSE"

import os.path as osp

import hydra
from omegaconf import DictConfig

from dcase24t6.train import train


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="test",
)
def test(cfg: DictConfig) -> None | float:
    return train(cfg)


if __name__ == "__main__":
    test()
