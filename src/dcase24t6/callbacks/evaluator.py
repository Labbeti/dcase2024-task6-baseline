#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from aac_metrics.classes.evaluate import Evaluate
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core.saving import save_hparams_to_yaml
from torch import Tensor
from torchoutil.nn.functional import move_to_rec

from dcase24t6.utils.collections import dict_list_to_list_dict
from dcase24t6.utils.dcase import export_to_csv_for_dcase_aac

logger = logging.getLogger(__name__)


EvalStage = Literal["val", "test", "predict"]
EVAL_STAGES = ("val", "test", "predict")


class Evaluator(Callback):
    def __init__(
        self,
        save_dir: str | Path,
        val_metrics: str | Iterable[str] = (),
        test_metrics: str | Iterable[str] = "all",
        exclude_keys: str | Iterable[str] | None = None,
    ) -> None:
        save_dir = Path(save_dir).resolve()

        if exclude_keys is None:
            exclude_keys = []
        elif isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        else:
            exclude_keys = list(exclude_keys)

        super().__init__()
        self.save_dir = save_dir
        self.exclude_keys = exclude_keys

        self.metrics = {
            "val": Evaluate(metrics=val_metrics),
            "test": Evaluate(metrics=test_metrics),
        }
        self.all_results: list[dict[str, Any]] = []

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

    def on_predict_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        return self._on_epoch_start("predict")

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

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_batch_end("predict", outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._on_epoch_end("val", pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_epoch_end("test", pl_module)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._on_epoch_end("predict", pl_module)

    # --- Private methods
    def _on_epoch_start(self, stage: EvalStage) -> None:
        self.all_results = [
            result for result in self.all_results if result["stage"] != stage
        ]

    def _on_batch_end(
        self,
        stage: EvalStage,
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

    def _on_epoch_end(self, stage: EvalStage, pl_module: LightningModule) -> None:
        stage_results = [
            result for result in self.all_results if result["stage"] == stage
        ]

        REQUIRED_KEYS = ("dataset", "subset", "fname", "candidates")
        if not all(key in result for result in stage_results for key in REQUIRED_KEYS):
            logger.info(
                f"Skipping stage results {stage} because it is missing required output keys."
            )
            return None

        dataset_names = [
            "{dataset}_{subset}".format(**result) for result in stage_results
        ]
        uniq_dataset_names = dict.fromkeys(dataset_names)

        for dataset_name in uniq_dataset_names:
            dataset_results = [
                result
                for name, result in zip(dataset_names, stage_results)
                if dataset_name == name
            ]

            match stage:
                case "val":
                    corpus_scores, _sentences_scores = self._get_results(
                        dataset_results, stage, dataset_name
                    )
                    pl_module.log_dict(corpus_scores, batch_size=len(dataset_results))

                case "test":
                    corpus_scores, sentences_scores = self._get_results(
                        dataset_results, stage, dataset_name
                    )
                    pl_module.log_dict(corpus_scores, batch_size=len(dataset_results))
                    self._save_results(
                        pl_module,
                        dataset_results,
                        corpus_scores,
                        sentences_scores,
                        stage,
                        dataset_name,
                    )
                    self._save_outputs(dataset_results, stage, dataset_name)

                case "predict":
                    self._save_outputs(dataset_results, stage, dataset_name)

                case invalid:
                    logger.warning(
                        f"Unknown stage '{invalid}'. (expected one of {EVAL_STAGES})"
                    )

    def _get_results(
        self,
        dataset_results: list[dict[str, Any]],
        stage: Literal["val", "test"],
        dataset_name: str,
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        candidates = [result["candidates"] for result in dataset_results]
        mult_references = [result["mult_references"] for result in dataset_results]

        corpus_scores, sentences_scores = self.metrics[stage](
            candidates, mult_references
        )
        corpus_scores = {
            f"{stage}/{dataset_name}.{k}": _tensor_to_builtin(v)
            for k, v in corpus_scores.items()
        }
        sentences_scores = {
            k: _tensor_to_builtin(v) for k, v in sentences_scores.items()
        }
        return corpus_scores, sentences_scores

    def _save_results(
        self,
        pl_module: LightningModule,
        dataset_results: list[dict[str, Any]],
        corpus_scores: dict[str, float],
        sentences_scores: dict[str, list[float]],
        stage: EvalStage,
        dataset_name: str,
    ) -> None:
        logger.info(
            f"Metrics results for {stage} at epoch {pl_module.current_epoch}:\n{yaml.dump(corpus_scores, sort_keys=False)}"
        )
        for pl_logger in pl_module.loggers:
            pl_logger.log_metrics(corpus_scores)

        corpus_scores_fpath = self.save_dir.joinpath(
            f"{stage}_{dataset_name}_scores.yaml"
        )
        save_hparams_to_yaml(corpus_scores_fpath, corpus_scores)

        sentences_scores_lst = dict_list_to_list_dict(sentences_scores)
        rows = [
            result | scores
            for result, scores in zip(dataset_results, sentences_scores_lst)
        ]
        rows = [
            {
                k: v
                for k, v in row.items()
                if all(re.search(pattern, k) for pattern in self.exclude_keys)
            }
            for row in rows
        ]
        rows = [{k: _tensor_to_builtin(v) for k, v in row.items()} for row in rows]
        fieldnames = rows[0].keys()

        sentences_scores_fpath = self.save_dir.joinpath(
            f"{stage}_{dataset_name}_outputs.csv"
        )
        with open(sentences_scores_fpath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_outputs(
        self,
        dataset_results: list[dict[str, Any]],
        stage: EvalStage,
        dataset_name: str,
    ) -> None:
        candidates = [result["candidates"] for result in dataset_results]
        fnames = [result["fname"] for result in dataset_results]

        submission_fpath = self.save_dir.joinpath(
            f"labbe_irit_task6_submission_1_{dataset_name}.csv"
        )
        export_to_csv_for_dcase_aac(
            submission_fpath,
            fnames,
            candidates,
        )


def _tensor_to_builtin(v: Any) -> Any:
    if not isinstance(v, Tensor):
        return v
    elif v.ndim == 0:
        return v.item()
    else:
        return v.tolist()
