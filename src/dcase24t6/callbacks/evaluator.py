#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Mapping

from aac_metrics.classes.evaluate import Evaluate
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from torchoutil.nn.functional import move_to_rec


class Evaluator(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.metrics = Evaluate(metrics="all")

        self.val_outputs = {}
        self.test_outputs = {}
        self.scores = {}

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.val_outputs = {}

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        val_candidates = self.val_outputs["candidates"]
        val_mult_references = self.val_outputs["mult_references"]
        self.evaluate(val_candidates, val_mult_references)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_outputs = {}

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        test_candidates = self.test_outputs["candidates"]
        test_mult_references = self.test_outputs["mult_references"]
        self.evaluate(test_candidates, test_mult_references)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, Mapping):
            return None

        outputs = move_to_rec(outputs, device="cpu")
        outputs = {k: list(v) for k, v in outputs.items()}

        if len(self.val_outputs) == 0:
            self.val_outputs = outputs
        else:
            for k, v in outputs.items():
                self.val_outputs[k] += v

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, Mapping):
            return None

        outputs = move_to_rec(outputs, device="cpu")
        outputs = {k: list(v) for k, v in outputs.items()}

        if len(self.val_outputs) == 0:
            self.test_outputs = outputs
        else:
            for k, v in outputs.items():
                self.test_outputs[k] += v

    def evaluate(
        self, candidates: list[str], mult_references: list[list[str]]
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        corpus_scores, sentences_scores = self.metrics(candidates, mult_references)
        corpus_scores = {k: v.item() for k, v in corpus_scores.items()}
        sentences_scores = {k: v.tolist() for k, v in sentences_scores.items()}
        return corpus_scores, sentences_scores
