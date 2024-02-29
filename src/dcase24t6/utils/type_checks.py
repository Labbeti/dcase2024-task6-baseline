#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, TypeGuard


def is_list_int(x: Any) -> TypeGuard[list[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)
