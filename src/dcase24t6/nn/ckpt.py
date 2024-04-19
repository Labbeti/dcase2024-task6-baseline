#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.hub.registry import RegistryHub

# Zenodo link : https://zenodo.org/records/8020843
# Hash type : md5
CNEXT_REGISTRY = RegistryHub(
    infos={
        "cnext_nobl": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1",
            "hash_value": "e069ecd1c7b880268331119521c549f2",
            "hash_type": "md5",
            "fname": "convnext_tiny_471mAP.pth",
            "state_dict_key": "model",
        },
        "cnext_bl_70": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
            "hash_value": "0688ae503f5893be0b6b71cb92f8b428",
            "hash_type": "md5",
            "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
            "state_dict_key": "model",
        },
        "cnext_bl_75": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/records/10987498/files/convnext_tiny_465mAP_BL_AC_75kit.pth?download=1",
            "hash_value": "f6f57c87b7eb664a23ae8cad26eccaa0",
            "hash_type": "md5",
            "fname": "convnext_tiny_465mAP_BL_AC_75kit.pth",
        },
    },
)

# Zenodo link : https://zenodo.org/records/10849427
# Hash type : md5
BASELINE_REGISTRY = RegistryHub(
    infos={
        "baseline_weights": {
            "architecture": "TransDecoderModel",
            "url": "https://zenodo.org/records/10849427/files/epoch_192-step_001544-mode_min-val_loss_3.3758.ckpt?download=1",
            "hash_value": "9514a8e6fa547bd01fb1badde81c6d10",
            "hash_type": "md5",
            "fname": "dcase2024-task6-baseline/checkpoints/best.ckpt",
            "state_dict_key": "state_dict",
        },
        "baseline_tokenizer": {
            "architecture": "AACTokenizer",
            "url": "https://zenodo.org/records/10849427/files/tokenizer.json?download=1",
            "hash_value": "ee3fef19f7d0891d820d84035483a900",
            "hash_type": "md5",
            "fname": "dcase2024-task6-baseline/tokenizer.json",
        },
    },
)
