#!/usr/bin/env python
# -*- coding: utf-8 -*-


__name__ = "dcase24t6"
__author__ = "Étienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Étienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.1.0"

import sys

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from dcase24t6.fit import fit
from dcase24t6.info import print_info


def print_usage() -> None:
    print(
        """
    Usage:
        dcase24t6 [SUBCOMMAND] [OPTIONS]

    Subcommands:
        fit \t Train a specified model.
        test \t Test a pretrained model.
        info \t Show installation information.
    """
    )


def main() -> None:
    if len(sys.argv) <= 1:
        return print_usage()

    target = sys.argv[1]
    hydra_args = sys.argv[2:]

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="train", overrides=hydra_args)

        match target:
            case "fit":
                return fit(cfg)
            case "info":
                return print_info()
            case _:
                return print_usage()
