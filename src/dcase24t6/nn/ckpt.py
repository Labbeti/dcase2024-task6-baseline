#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.utils.ckpt import ModelCheckpointRegister

# Zenodo link : https://zenodo.org/records/8020843
# Hash type : md5
CNEXT_REGISTER = ModelCheckpointRegister(
    infos={
        "cnext_bl_70": {
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

# Zenodo link : https://zenodo.org/records/10849427
# Hash type : md5
BASELINE_REGISTER = ModelCheckpointRegister(
    infos={
        "baseline_weights": {
            "architecture": "TransDecoderModel",
            "url": "https://zenodo.org/records/10849427/files/epoch_192-step_001544-mode_min-val_loss_3.3758.ckpt?download=1",
            "hash": "9514a8e6fa547bd01fb1badde81c6d10",
            "fname": "dcase2024-task6-baseline/checkpoints/best.ckpt",
        },
        "baseline_tokenizer": {
            "architecture": "AACTokenizer",
            "url": "https://zenodo.org/records/10849427/files/tokenizer.json?download=1",
            "hash": "ee3fef19f7d0891d820d84035483a900",
            "fname": "dcase2024-task6-baseline/tokenizer.json",
        },
    },
    state_dict_key="state_dict",
)
