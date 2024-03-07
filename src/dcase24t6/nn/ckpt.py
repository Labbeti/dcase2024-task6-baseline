#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.utils.ckpt import ModelCheckpointRegister

# Zenodo link : https://zenodo.org/record/8020843
# Hash type : md5
CNEXT_REGISTER = ModelCheckpointRegister(
    infos={
        "cnext_bl": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
            "hash": "0688ae503f5893be0b6b71cb92f8b428",
            "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
        },
        "cnext_nobl": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1",
            "hash": "e069ecd1c7b880268331119521c549f2",
            "fname": "convnext_tiny_471mAP.pth",
        },
    },
    state_dict_key="model",
)
