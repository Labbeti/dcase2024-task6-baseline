#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
from logging import FileHandler
from pathlib import Path

logger = logging.getLogger(__name__)


class CustomFileHandler(FileHandler):
    """FileHandler which builds intermediate directories.

    Used for export hydra logs to a file contained in a folder that does not exists yet at the start of the program.
    """

    def __init__(
        self,
        filename: str | Path,
        mode: str = "a",
        encoding: str | None = None,
        delay: bool = True,
    ) -> None:
        filename = Path(filename)
        parent_dpath = osp.dirname(filename)
        if parent_dpath != "":
            try:
                os.makedirs(parent_dpath, exist_ok=True)
            except PermissionError as err:
                logger.warning(
                    f"Cannot create intermediate directories for hydra log files. (with {err=})"
                )
        super().__init__(filename, mode, encoding, delay)
