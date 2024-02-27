#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


pylog = logging.getLogger(__name__)


class Compose(Generic[T, U]):
    def __init__(self, *fns: Callable[[T], U]) -> None:
        super().__init__()
        self.fns = fns

    def __call__(self, x: T) -> U:
        for fn in self.fns:
            x = fn(x)
        return x  # type: ignore
