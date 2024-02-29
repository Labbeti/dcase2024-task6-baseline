#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, TypedDict

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torchoutil import (
    lengths_to_pad_mask,
    masked_mean,
    randperm_diff,
    tensor_to_pad_mask,
)
from torchoutil.nn import Transpose

from dcase24t6.augmentations.mixup import sample_lambda
from dcase24t6.datamodules.hdf import Stage
from dcase24t6.models.aac import AACModel
from dcase24t6.nn.decoders.aac_tfmer import AACTransformerDecoder
from dcase24t6.nn.decoding.beam import generate
from dcase24t6.nn.decoding.common import get_forbid_rep_mask_content_words
from dcase24t6.nn.decoding.forcing import teacher_forcing
from dcase24t6.nn.decoding.greedy import greedy_search
from dcase24t6.optim.schedulers import CosDecayScheduler
from dcase24t6.optim.utils import create_params_groups
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


class Batch(TypedDict):
    frame_embs: Tensor
    frame_embs_shape: Tensor


class TrainBatch(Batch):
    captions: Tensor


class ValBatch(Batch):
    mult_captions: Tensor
    mult_references: list[list[str]]
    dataset: str
    subset: str
    fname: str


TestBatch = ValBatch


class PredictBatch(Batch):
    dataset: str
    subset: str
    fname: str


ModelOutput = dict[str, Tensor]


class AudioEncoding(TypedDict):
    frame_embs: Tensor
    frame_embs_pad_mask: Tensor


class TransDecoderModel(AACModel):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        # Model architecture args
        in_features: int = 768,
        d_model: int = 256,
        # Train args
        label_smoothing: float = 0.2,
        mixup_alpha: float = 0.4,
        # Inference args
        min_pred_size: int = 3,
        max_pred_size: int = 20,
        beam_size: int = 3,
        # Optimizer args
        custom_weight_decay: bool = True,
        lr: float = 5e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 2.0,
        sched_num_steps: int = 400,
        # Other args
        verbose: int = 0,
    ) -> None:
        super().__init__(tokenizer)
        self.projection: nn.Module = nn.Identity()
        self.decoder: AACTransformerDecoder = None  # type: ignore
        self.save_hyperparameters(ignore=["tokenizer"])

    def is_built(self) -> bool:
        return self.decoder is not None

    def setup(self, stage: Stage) -> None:
        if not self.is_built():
            projection = nn.Sequential(
                nn.Dropout(p=0.5),
                Transpose(1, 2),
                nn.Linear(self.hparams["in_features"], self.hparams["d_model"]),
                nn.ReLU(inplace=True),
                Transpose(1, 2),
                nn.Dropout(p=0.5),
            )
            decoder = AACTransformerDecoder(
                vocab_size=self.tokenizer.get_vocab_size(),
                pad_id=self.tokenizer.pad_token_id,
                d_model=self.hparams["d_model"],
            )

            forbid_rep_mask = get_forbid_rep_mask_content_words(
                vocab_size=self.tokenizer.get_vocab_size(),
                token_to_id=self.tokenizer.get_token_to_id(),
                device=self.device,
                verbose=self.hparams["verbose"],
            )

            self.projection = projection
            self.decoder = decoder
            self.register_buffer("forbid_rep_mask", forbid_rep_mask)
            self.forbid_rep_mask: Optional[Tensor]

        if stage in ("fit", None):
            batch = next(iter(self.datamodule.train_dataloader()))
            self.example_input_array = {"batch": batch}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.hparams["custom_weight_decay"]:
            params = create_params_groups(self, self.hparams["weight_decay"])
        else:
            params = self.parameters()

        optimizer_args = {
            name: self.hparams[name] for name in ("lr", "betas", "eps", "weight_decay")
        }
        optimizer = AdamW(params, **optimizer_args)

        num_steps = self.hparams["sched_num_steps"]
        scheduler = CosDecayScheduler(optimizer, num_steps)

        return [optimizer], [scheduler]

    def training_step(self, batch: TrainBatch) -> Tensor:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        captions = batch["captions"]

        bsize, _max_caption_length = captions.shape
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        del captions

        indexes = randperm_diff(bsize, device=self.device)
        audio, audio_shape, lbd = self.mix_audio(audio, audio_shape, indexes)
        captions_in_pad_mask = tensor_to_pad_mask(
            captions_in, pad_value=self.tokenizer.pad_token_id
        )
        captions_in = self.input_emb_layer(captions_in)
        captions_in = captions_in * lbd + captions_in[indexes] * (1.0 - lbd)

        encoded = self.encode_audio(audio, audio_shape)
        decoded = self.decode_audio(
            encoded,
            captions=captions_in,
            captions_pad_mask=captions_in_pad_mask,
            method="forcing",
        )
        logits = decoded["logits"]

        loss = self.train_criterion(logits, captions_out)
        self.log("train/loss", loss, batch_size=bsize, prog_bar=True)

        return loss

    def validation_step(self, batch: ValBatch) -> dict[str, Tensor]:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        mult_captions = batch["mult_captions"]

        bsize, max_captions_per_audio, _max_caption_length = mult_captions.shape
        mult_captions_in = mult_captions[:, :, :-1]
        mult_captions_out = mult_captions[:, :, 1:]
        is_valid_caption = (mult_captions != self.tokenizer.pad_token_id).any(dim=2)
        del mult_captions

        encoded = self.encode_audio(audio, audio_shape)
        losses = torch.empty(
            (
                bsize,
                max_captions_per_audio,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(max_captions_per_audio):
            captions_in_i = mult_captions_in[:, i]
            captions_out_i = mult_captions_out[:, i]

            decoded_i = self.decode_audio(encoded, captions=captions_in_i)
            logits_i = decoded_i["logits"]
            losses_i = self.val_criterion(logits_i, captions_out_i)
            losses[:, i] = losses_i

        loss = masked_mean(losses, is_valid_caption)
        self.log("val/loss", loss, batch_size=bsize, prog_bar=True)

        decoded = self.decode_audio(encoded, method="generate")
        outputs = {
            "loss": losses,
        } | decoded
        return outputs

    def test_step(self, batch: TestBatch) -> dict[str, Any]:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        mult_captions = batch["mult_captions"]

        bsize, max_captions_per_audio, _max_caption_length = mult_captions.shape
        mult_captions_in = mult_captions[:, :, :-1]
        mult_captions_out = mult_captions[:, :, 1:]
        is_valid_caption = (mult_captions != self.tokenizer.pad_token_id).any(dim=2)
        del mult_captions

        encoded = self.encode_audio(audio, audio_shape)
        losses = torch.empty(
            (
                bsize,
                max_captions_per_audio,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(max_captions_per_audio):
            captions_in_i = mult_captions_in[:, i]
            captions_out_i = mult_captions_out[:, i]

            decoded_i = self.decode_audio(encoded, captions=captions_in_i)
            logits_i = decoded_i["logits"]
            losses_i = self.test_criterion(logits_i, captions_out_i)
            losses[:, i] = losses_i

        loss = masked_mean(losses, is_valid_caption)
        self.log("test/loss", loss, batch_size=bsize, prog_bar=True)

        decoded = self.decode_audio(encoded, method="generate")
        outputs = {
            "loss": losses,
        } | decoded
        return outputs

    def forward(
        self,
        batch: Batch,
        **method_kwargs,
    ) -> ModelOutput:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        captions = batch.get("captions", None)
        encoded = self.encode_audio(audio, audio_shape)
        decoded = self.decode_audio(encoded, captions, **method_kwargs)
        return decoded

    def train_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=self.hparams["label_smoothing"],
        )
        return loss

    def val_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        losses = F.cross_entropy(
            logits,
            target,
            ignore_index=self.tokenizer.pad_token_id,
            reduction="none",
        )
        # We apply mean only on second dim to get a tensor of shape (bsize,)
        losses = masked_mean(losses, target != self.tokenizer.pad_token_id, dim=1)
        return losses

    def test_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.val_criterion(logits, target)

    def input_emb_layer(self, ids: Tensor) -> Tensor:
        return self.decoder.emb_layer(ids)

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

    def encode_audio(
        self, frame_embs: Tensor, frame_embs_shape: Tensor
    ) -> AudioEncoding:
        # frame_embs: (bsize, 1, in_features, max_seq_size)
        # frame_embs_shape: (bsize, 3)

        time_dim = -1

        frame_embs = frame_embs.squeeze(dim=1)  # remove channel dim
        frame_embs_lens = frame_embs_shape[:, time_dim]
        # frame_embs_lens: (bsize,)

        frame_embs = self.projection(frame_embs)
        # frame_embs: (bsize, d_model, max_seq_size)

        frame_embs_max_len = max(
            int(frame_embs_lens.max().item()), frame_embs.shape[time_dim]
        )
        frame_embs_pad_mask = lengths_to_pad_mask(frame_embs_lens, frame_embs_max_len)

        return {
            "frame_embs": frame_embs,
            "frame_embs_pad_mask": frame_embs_pad_mask,
        }

    def decode_audio(
        self,
        audio_encoding: AudioEncoding,
        captions: Optional[Tensor] = None,
        captions_pad_mask: Optional[Tensor] = None,
        method: str = "auto",
        **method_overrides,
    ) -> dict[str, Tensor]:
        if method == "auto":
            if captions is None:
                method = "generate"
            else:
                method = "forcing"

        common_args: dict[str, Any] = {
            "decoder": self.decoder,
            "pad_id": self.tokenizer.pad_token_id,
        } | audio_encoding

        match method:
            case "forcing":
                forcing_args = {
                    "caps_in": captions,
                    "caps_in_pad_mask": captions_pad_mask,
                }
                kwargs = common_args | forcing_args | method_overrides
                logits = teacher_forcing(**kwargs)
                outs = {"logits": logits}

            case "greedy":
                greedy_args = {
                    "bos_id": self.tokenizer.bos_token_id,
                    "eos_id": self.tokenizer.eos_token_id,
                    "vocab_size": self.tokenizer.get_vocab_size(),
                    "min_pred_size": self.hparams["min_pred_size"],
                    "max_pred_size": self.hparams["max_pred_size"],
                    "forbid_rep_mask": self.forbid_rep_mask,
                }
                kwargs = common_args | greedy_args | method_overrides
                logits = greedy_search(**kwargs)
                outs = {"logits": logits}

            case "generate":
                generate_args = {
                    "bos_id": self.tokenizer.bos_token_id,
                    "eos_id": self.tokenizer.eos_token_id,
                    "vocab_size": self.tokenizer.get_vocab_size(),
                    "min_pred_size": self.hparams["min_pred_size"],
                    "max_pred_size": self.hparams["max_pred_size"],
                    "forbid_rep_mask": self.forbid_rep_mask,
                    "beam_size": self.hparams["beam_size"],
                }
                kwargs = common_args | generate_args | method_overrides
                outs = generate(**kwargs)
                outs = outs._asdict()

                # Decode predictions ids to sentences
                keys = list(outs.keys())
                for key in keys:
                    if "prediction" not in key:
                        continue
                    preds: Tensor = outs[key]
                    if preds.ndim == 2:
                        cands = self.tokenizer.decode_batch(preds.tolist())
                    elif preds.ndim == 3:
                        cands = [
                            self.tokenizer.decode_batch(value_i)
                            for value_i in preds.tolist()
                        ]
                    else:
                        raise ValueError(
                            f"Invalid shape {preds.ndim=}. (expected one of {(2, 3)})"
                        )
                    new_key = key.replace("prediction", "candidate")
                    outs[new_key] = cands

            case method:
                DECODE_METHODS = ("forcing", "greedy", "generate", "auto")
                raise ValueError(
                    f"Unknown argument {method=}. (expected one of {DECODE_METHODS})"
                )

        return outs
