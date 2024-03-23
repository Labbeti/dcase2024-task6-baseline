#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
import subprocess
from subprocess import CalledProcessError
from typing import TypeVar

T = TypeVar("T")
pylog = logging.getLogger(__name__)


def get_git_hash(
    cwd: str = osp.dirname(__file__),
    default: T = "UNKNOWN",
) -> str | T:
    """
    Return the current git hash in the current directory.

    :returns: The git hash. If an error occurs, returns 'UNKNOWN'.
    """
    try:
        git_hash = subprocess.check_output("git describe --always".split(" "), cwd=cwd)
        git_hash = git_hash.decode("UTF-8").replace("\n", "")
        return git_hash
    except (CalledProcessError, PermissionError) as err:
        pylog.warning(
            f"Cannot get current git hash from {cwd=}. (error message: '{err}')"
        )
        return default
