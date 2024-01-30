#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import yaml
from aac_metrics.classes.evaluate import Evaluate
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback


class Evaluator(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.metrics = Evaluate(metrics="all")

        self.val_candidates = []
        self.val_mult_references = []
        self.test_candidates = []
        self.test_mult_references = []

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.val_candidates = []
        self.val_mult_references = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.evaluate(self.val_candidates, self.val_mult_references)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_candidates = []
        self.test_mult_references = []

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate(self.test_candidates, self.test_mult_references)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        candidates = outputs["candidates"]
        mult_references = outputs["mult_references"]
        self.val_candidates += candidates
        self.val_mult_references += mult_references

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        candidates = outputs["candidates"]
        mult_references = outputs["mult_references"]
        self.test_candidates += candidates
        self.test_mult_references += mult_references

    def evaluate(self, candidates: list[str], mult_references: list[list[str]]) -> None:
        corpus_scores, sentences_scores = self.metrics(candidates, mult_references)
        corpus_scores = {k: v.tolist() for k, v in corpus_scores.items()}
        sentences_scores = {k: v.item() for k, v in sentences_scores.items()}

        print(yaml.dump(corpus_scores, sort_keys=False))
