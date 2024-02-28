#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from aac_metrics.classes.evaluate import Evaluate
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from torchoutil.nn.functional import move_to_rec

from dcase24t6.utils.collections import dict_list_to_list_dict

logger = logging.getLogger(__name__)


class Evaluator(Callback):
    def __init__(self, logdir: str | Path) -> None:
        logdir = Path(logdir).resolve()
        super().__init__()
        self.metrics = Evaluate(metrics="all")

        self.all_results = []
        self.corpus_scores = {}

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        return self._on_epoch_start("val")

    def on_test_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        return self._on_epoch_start("test")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_batch_end("val", outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_batch_end("test", outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._on_epoch_end("val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_epoch_end("test")

    def _on_epoch_start(self, stage: Literal["val", "test"]) -> None:
        self.all_results = [
            result for result in self.all_results if result["stage"] != "val"
        ]

    def _on_batch_end(
        self,
        stage: Literal["val", "test"],
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict) or not isinstance(batch, dict):
            return None

        batch_size = len(next(iter(outputs.values())))

        result_dict = outputs | batch
        result_dict = move_to_rec(result_dict, device="cpu")
        result_dict |= {
            "dataloader_idx": [dataloader_idx] * batch_size,
            "batch_idx": [batch_idx] * batch_size,
            "stage": [stage] * batch_size,
        }
        result = dict_list_to_list_dict(result_dict)
        self.all_results += result

    def _on_epoch_end(self, stage: str) -> None:
        stage_results = [
            result for result in self.all_results if result["stage"] == stage
        ]
        full_names = ["{dataset}_{subset}".format(**result) for result in stage_results]
        uniq_full_names = dict.fromkeys(full_names)

        for uniq_name in uniq_full_names:
            dataset_results = [
                result
                for name, result in zip(full_names, stage_results)
                if uniq_name == name
            ]
            try:
                candidates = [result["candidates"] for result in dataset_results]
            except KeyError as err:
                # breakpoint()
                raise err
            mult_references = [result["mult_references"] for result in dataset_results]
            corpus_scores, sentences_scores = self.evaluate(
                uniq_name, candidates, mult_references
            )

            if stage == "test":
                logger.info(f"{yaml.dump(corpus_scores, sort_keys=False)}")

    def evaluate(
        self,
        dataset_name: str,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        corpus_scores, sentences_scores = self.metrics(candidates, mult_references)
        corpus_scores = {
            f"{dataset_name}.{k}": v.item() for k, v in corpus_scores.items()
        }
        sentences_scores = {
            f"{dataset_name}.{k}": v.tolist() for k, v in sentences_scores.items()
        }
        return corpus_scores, sentences_scores
