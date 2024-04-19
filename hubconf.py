#!/usr/bin/env python
# -*- coding: utf-8 -*-

dependencies = ["git+https://github.com/Labbeti/dcase2024-task6-baseline"]


from dcase24t6.nn.hub import baseline_pipeline

__all__ = ["baseline_pipeline"]
