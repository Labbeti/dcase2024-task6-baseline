#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torchoutil import masked_mean, randperm_diff, tensor_to_pad_mask

from dcase24t6.models.aac import AACModel
from dcase24t6.optim.utils import create_params_groups
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer
from dcase24t6.transforms.mixup import sample_lambda

Batch = dict
TrainBatch = dict
ValBatch = dict
TestBatch = dict
PredictBatch = dict
Encoded = dict
Decoded = dict


class TransDecoderModel(AACModel):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        # Model args
        # Train args
        label_smoothing: float = 0.2,
        mixup_alpha: float = 0.4,
        # Optimizer args
        custom_weight_decay: bool = True,
        lr: float = 5e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 2.0,
    ) -> None:
        super().__init__(tokenizer)
        self.save_hyperparameters(ignore=["tokenizer"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.hparams["custom_weight_decay"]:
            params = create_params_groups(self, self.hparams["weight_decay"])
        else:
            params = self.parameters()

        optimizer_args = {
            name: self.hparams[name] for name in ("lr", "betas", "eps", "weight_decay")
        }
        optimizer = AdamW(params, **optimizer_args)
        return optimizer

    def training_step(self, batch: TrainBatch) -> Tensor:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        captions = batch["captions"]

        batch_size, _max_caption_length = captions.shape
        captions_in = captions[:, :, :-1]
        captions_out = captions[:, :, 1:]
        del captions

        indexes = randperm_diff(batch_size, device=self.device)
        audio, audio_shape, lbd = self.mix_audio(audio, audio_shape, indexes)
        captions_in_pad_mask = tensor_to_pad_mask(captions_in, pad_value=self.pad_id)
        captions_in = self.input_emb_layer(captions_in)
        captions_in = captions_in * lbd + captions_in[indexes] * (1.0 - lbd)

        encoded = self.encode_audio(audio, audio_shape)
        decoded = self.decode_audio(
            encoded, captions=captions_in, captions_pad_mask=captions_in_pad_mask
        )
        logits = decoded["logits"]

        loss = self.train_criterion(logits, captions_out)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: ValBatch) -> None:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        mult_captions = batch["mult_captions"]

        batch_size, max_captions_per_audio, _max_caption_length = mult_captions.shape
        mult_captions_in = mult_captions[:, :, :-1]
        mult_captions_out = mult_captions[:, :, 1:]
        is_valid_caption = (mult_captions != self.pad_id).any(dim=2)
        del mult_captions

        encoded = self.encode_audio(audio, audio_shape)
        losses = torch.empty(
            (
                batch_size,
                max_captions_per_audio,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(mult_captions_in.shape[1]):
            captions_in_i = mult_captions_in[:, i]
            captions_out_i = mult_captions_out[:, i]

            decoded_i = self.decode_audio(encoded, captions=captions_in_i)
            logits_i = decoded_i["logits"]
            losses_i = self.val_criterion(logits_i, captions_out_i)
            losses[:, i] = losses_i

        loss = masked_mean(losses, is_valid_caption)
        self.log("val/loss", loss)

    def test_step(self, batch: TestBatch) -> dict[str, Any]:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        mult_captions = batch["mult_captions"]

        batch_size, max_captions_per_audio, _max_caption_length = mult_captions.shape
        mult_captions_in = mult_captions[:, :, :-1]
        mult_captions_out = mult_captions[:, :, 1:]
        is_valid_caption = (mult_captions != self.pad_id).any(dim=2)
        del mult_captions

        encoded = self.encode_audio(audio, audio_shape)
        losses = torch.empty(
            (
                batch_size,
                max_captions_per_audio,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(mult_captions_in.shape[1]):
            captions_in_i = mult_captions_in[:, i]
            captions_out_i = mult_captions_out[:, i]

            decoded_i = self.decode_audio(encoded, captions=captions_in_i)
            logits_i = decoded_i["logits"]
            losses_i = self.test_criterion(logits_i, captions_out_i)
            losses[:, i] = losses_i

        loss = masked_mean(losses, is_valid_caption)
        self.log("test/loss", loss)

        output = {
            "loss": losses,
        }
        return output

    def forward(
        self,
        batch: Batch,
        **method_kwargs,
    ) -> Decoded:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        captions = batch.get("captions", None)
        encoded = self.encode_audio(audio, audio_shape)
        decoded = self.decode_audio(encoded, captions, **method_kwargs)
        return decoded

    def train_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.pad_id,
            label_smoothing=self.hparams["label_smoothing"],
        )
        return loss

    def val_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.pad_id,
        )
        return loss

    def test_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        losses = F.cross_entropy(
            logits,
            target,
            ignore_index=self.pad_id,
            reduction="none",
        )
        losses = masked_mean(losses, target != self.pad_id, dim=1)
        return losses

    def input_emb_layer(self, ids: Tensor) -> Tensor:
        raise NotImplementedError

    def mix_audio(
        self, audio: Tensor, audio_shape: Tensor, indexes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        lbd = sample_lambda(
            self.hparams["mixup_alpha"],
            asymmetric=True,
            size=(),
        )
        mixed_audio = audio * lbd + audio[indexes] * (1.0 - lbd)
        mixed_audio_shape = torch.max(audio_shape, audio_shape[indexes])
        return mixed_audio, mixed_audio_shape, lbd

    def encode_audio(self, audio: Tensor, audio_shape: Tensor) -> Encoded:
        raise NotImplementedError

    def decode_audio(
        self,
        encoded: Encoded,
        captions: Optional[Tensor] = None,
        captions_pad_mask: Optional[Tensor] = None,
        method: str = "auto",
        **method_kwargs,
    ) -> Decoded:
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        raise NotImplementedError

    @property
    def bos_id(self) -> int:
        raise NotImplementedError

    @property
    def eos_id(self) -> int:
        raise NotImplementedError

    @property
    def unk_id(self) -> int:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
