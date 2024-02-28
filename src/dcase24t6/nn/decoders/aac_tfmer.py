#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from dcase24t6.nn.functional import get_activation_fn
from dcase24t6.nn.modules import PositionalEncoding

pylog = logging.getLogger(__name__)


class AACTransformerDecoder(nn.TransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        acti_name: Union[str, Callable] = "gelu",
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        emb_scale_grad_by_freq: bool = False,
        layer_norm_eps: float = 1e-5,
        nhead: int = 8,
        num_decoder_layers: int = 6,
    ) -> None:
        if isinstance(acti_name, str):
            activation = get_activation_fn(acti_name)
        else:
            activation = acti_name

        pos_encoding = PositionalEncoding(d_model, dropout)
        emb_layer = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
            scale_grad_by_freq=emb_scale_grad_by_freq,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=False,
            norm_first=False,
        )
        classifier = nn.Linear(d_model, vocab_size)

        super().__init__(decoder_layer, num_decoder_layers)

        # Hparams
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # Layers
        self.emb_layer = emb_layer
        self.pos_encoding = pos_encoding
        self.classifier = classifier

    def forward(
        self,
        frame_embs: Tensor,
        frame_embs_attn_mask: Optional[Tensor],
        frame_embs_pad_mask: Optional[Tensor],
        caps_in: Tensor,
        caps_in_attn_mask: Optional[Tensor],
        caps_in_pad_mask: Optional[Tensor],
    ) -> Tensor:
        """
        :param frame_embs: (n_frames, bsize, d_model)
        :param caps_in: (caps_in_len, bsize) or (caps_in_len, bsize, d_model)
        :param frame_embs_attn_mask: (caps_in_len, n_frames) or None
        :param caps_in_attn_mask: (caps_in_len, caps_in_len)
        :param frame_embs_pad_mask: (bsize, n_frames) or None
        :param caps_in_pad_mask: (caps_in_len, bsize) or None
        :returns: logits of shape (caps_in_len, bsize, vocab_size)
        """
        self._check_args(
            frame_embs,
            frame_embs_pad_mask,
            frame_embs_attn_mask,
            caps_in,
            caps_in_pad_mask,
            caps_in_attn_mask,
        )
        caps_in_len = caps_in.shape[0]
        num_frames = frame_embs.shape[0]

        if frame_embs_pad_mask is not None and frame_embs_attn_mask is None:
            frame_embs_attn_mask = torch.zeros(
                (caps_in_len, num_frames),
                dtype=torch.bool,
                device=frame_embs.device,
            )

        if not caps_in.is_floating_point():
            caps_in = self.emb_layer(caps_in)

        # caps_in: (caps_in_len, bsize, d_model)
        d_model = caps_in.shape[-1]
        caps_in = caps_in * math.sqrt(d_model)
        caps_in = self.pos_encoding(caps_in)

        tok_embs_outs = super().forward(
            memory=frame_embs,
            memory_key_padding_mask=frame_embs_pad_mask,
            memory_mask=frame_embs_attn_mask,
            tgt=caps_in,
            tgt_key_padding_mask=caps_in_pad_mask,
            tgt_mask=caps_in_attn_mask,
        )
        tok_logits_out = self.classifier(tok_embs_outs)

        return tok_logits_out

    def _check_args(
        self,
        frame_embs: Tensor,
        frame_embs_pad_mask: Optional[Tensor],
        frame_embs_attn_mask: Optional[Tensor],
        caps_in: Tensor,
        caps_in_pad_mask: Optional[Tensor],
        caps_in_attn_mask: Optional[Tensor],
    ) -> None:
        if frame_embs.ndim != 3:
            raise ValueError(f"Invalid argument ndim {frame_embs.ndim=} (expected 3)")

        if caps_in.is_floating_point():
            if caps_in.ndim != 3:
                raise ValueError(f"Invalid argument ndim {caps_in.ndim=} (expected 3)")
        else:
            if caps_in.ndim != 2:
                raise ValueError(f"Invalid argument ndim {caps_in.ndim=} (expected 2)")

        if frame_embs_pad_mask is not None and frame_embs_pad_mask.ndim != 2:
            raise ValueError(
                f"Invalid argument ndim {frame_embs_pad_mask.ndim=} (expected 2)"
            )

        if frame_embs_attn_mask is not None and frame_embs_attn_mask.ndim != 2:
            raise ValueError(
                f"Invalid argument ndim {frame_embs_attn_mask.ndim=} (expected 2)"
            )

        if caps_in_pad_mask is not None and caps_in_pad_mask.ndim != 2:
            raise ValueError(
                f"Invalid argument ndim {caps_in_pad_mask.ndim=} (expected 2)"
            )

        if caps_in_attn_mask is not None and caps_in_attn_mask.ndim != 2:
            raise ValueError(
                f"Invalid argument ndim {caps_in_attn_mask.ndim=} (expected 2)"
            )
