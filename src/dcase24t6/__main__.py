#!/usr/bin/env python
# -*- coding: utf-8 -*-


def print_usage() -> None:
    print(
        """
    Usage:
        dcase24t6-[SUBCOMMAND] [OPTIONS]

    Subcommands:
        train \t Train a model.
        test \t Test a pretrained model.
        prepare \t Install data, metrics and models.
        info \t Show installation information.
    """
    )


if __name__ == "__main__":
    print_usage()
