#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import torch
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchoutil.utils.data.collate import AdvancedCollateDict
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import EmptyDataset, TransformWrapper
from torchoutil.utils.hdf import HDFDataset

from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

Stage = Literal["fit", "validate", "test", "predict"] | None
ALL_STAGES = ("fit", "validate", "test", "predict")


class HDFDatamodule(LightningDataModule):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        root: str | Path,
        train_hdfs: str | Iterable[str] = (),
        val_hdfs: str | Iterable[str] = (),
        test_hdfs: str | Iterable[str] = (),
        predict_hdfs: str | Iterable[str] = (),
        train_batch_keys: Iterable[str] | None = None,
        val_batch_keys: Iterable[str] | None = None,
        test_batch_keys: Iterable[str] | None = None,
        predict_batch_keys: Iterable[str] | None = None,
        # DataLoader args
        batch_size: int = 32,
        num_workers: int | Literal["auto"] = "auto",
        pin_memory: bool = True,
        train_drop_last: bool = False,
        # Other args
        verbose: int = 0,
    ) -> None:
        root = Path(root)
        if isinstance(train_hdfs, str):
            train_hdfs = [train_hdfs]
        else:
            train_hdfs = list(train_hdfs)

        if isinstance(val_hdfs, str):
            val_hdfs = [val_hdfs]
        else:
            val_hdfs = list(val_hdfs)

        if isinstance(test_hdfs, str):
            test_hdfs = [test_hdfs]
        else:
            test_hdfs = list(test_hdfs)

        if isinstance(predict_hdfs, str):
            predict_hdfs = [predict_hdfs]
        else:
            predict_hdfs = list(predict_hdfs)

        if num_workers == "auto":
            num_workers = get_auto_num_cpus()

        super().__init__()
        self.tokenizer = tokenizer
        self.root = root
        self.train_hdfs = train_hdfs
        self.val_hdfs = val_hdfs
        self.test_hdfs = test_hdfs
        self.predict_hdfs = predict_hdfs

        self.train_batch_keys = train_batch_keys
        self.val_batch_keys = val_batch_keys
        self.test_batch_keys = test_batch_keys
        self.predict_batch_keys = predict_batch_keys

        self.train_dataset = EmptyDataset()
        self.val_datasets = {}
        self.test_datasets = {}
        self.predict_datasets = {}

        self.train_collate_fn = None
        self.val_collate_fn = None
        self.test_collate_fn = None
        self.predict_collate_fn = None

        self.save_hyperparameters(ignore=["tokenizer"])

    def setup(self, stage: Stage = None) -> None:
        match stage:
            case "fit":
                self.setup_train()
                self.setup_val()
            case "validate":
                self.setup_train()
                self.setup_val()
            case "test":
                self.setup_test()
            case "predict":
                self.setup_predict()
            case None:
                self.setup_train()
                self.setup_val()
                self.setup_test()
                self.setup_predict()

    def teardown(self, stage: Stage = None) -> None:
        match stage:
            case "fit":
                self.teardown_train()
                self.teardown_val()
            case "validate":
                pass
            case "test":
                self.teardown_test()
            case "predict":
                self.teardown_predict()
            case None:
                self.teardown_train()
                self.teardown_val()
                self.teardown_test()
                self.teardown_predict()

    def train_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        refs = item["captions"]
        ref = random.choice(refs)
        caption = self.tokenizer.encode(ref, disable_unk_token=True)
        caption = torch.as_tensor(caption.ids)

        item["captions"] = caption
        item["references"] = ref

        if self.train_batch_keys is not None:
            item = {k: item[k] for k in self.train_batch_keys}
        return item

    def val_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._common_transform(item, self.val_batch_keys)

    def test_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._common_transform(item, self.test_batch_keys)

    def predict_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._common_transform(item, self.predict_batch_keys)

    def _common_transform(
        self,
        item: dict[str, Any],
        keys: Iterable[str] | None,
    ) -> dict[str, Any]:
        if "captions" in item:
            refs = item.pop("captions")
            captions = self.tokenizer.encode_batch(refs)
            captions = torch.as_tensor([cap.ids for cap in captions])

            item["mult_captions"] = captions
            item["mult_references"] = refs

        if keys is not None:
            item = {k: item[k] for k in keys}

        return item

    def setup_train(self) -> None:
        hdf_fnames = self.train_hdfs

        datasets = {}
        flat_references = []
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.root.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)

            # note: get raw captions before transform
            flat_references += [ref for refs in dataset[:, "captions"] for ref in refs]

            dataset = TransformWrapper(dataset, self.train_transform)
            datasets[hdf_fname] = dataset

        dataset = ConcatDataset(datasets.values())

        pad_values = {
            "frame_embs": 0.0,
            "captions": self.tokenizer.pad_token_id,
        }
        train_collate_fn = AdvancedCollateDict(pad_values)

        self.train_dataset = dataset
        self.train_collate_fn = train_collate_fn
        self.tokenizer.train_from_iterator(flat_references)

    def setup_val(self) -> None:
        datasets, collate_fn = self._common_setup(
            self.val_hdfs,
            self.val_transform,
        )
        self.val_datasets = datasets
        self.val_collate_fn = collate_fn

    def setup_test(self) -> None:
        datasets, collate_fn = self._common_setup(
            self.test_hdfs,
            self.test_transform,
        )
        self.test_datasets = datasets
        self.test_collate_fn = collate_fn

    def setup_predict(self) -> None:
        datasets, collate_fn = self._common_setup(
            self.predict_hdfs,
            self.predict_transform,
        )
        self.predict_datasets = datasets
        self.predict_collate_fn = collate_fn

    def _common_setup(
        self,
        hdf_fnames: list[str],
        transform: Callable | None,
    ) -> tuple[dict, Callable]:
        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.root.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            dataset = TransformWrapper(dataset, transform)
            datasets[hdf_fname] = dataset

        pad_values = {
            "frame_embs": 0.0,
            "mult_captions": self.tokenizer.pad_token_id,
        }
        collate_fn = AdvancedCollateDict(pad_values)
        return datasets, collate_fn

    def teardown_train(self) -> None:
        self.train_dataset = EmptyDataset()
        self.train_collate_fn = None

    def teardown_val(self) -> None:
        self.val_datasets = {}
        self.val_collate_fn = None

    def teardown_test(self) -> None:
        self.test_datasets = {}
        self.test_collate_fn = None

    def teardown_predict(self) -> None:
        self.predict_datasets = {}
        self.predict_collate_fn = None

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.train_collate_fn,
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["train_drop_last"],
            shuffle=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return self._get_dataloaders(self.val_datasets.values(), self.val_collate_fn)

    def test_dataloader(self) -> list[DataLoader]:
        return self._get_dataloaders(self.test_datasets.values(), self.test_collate_fn)

    def predict_dataloader(self) -> list[DataLoader]:
        return self._get_dataloaders(
            self.predict_datasets.values(), self.predict_collate_fn
        )

    def _get_dataloaders(
        self,
        datasets: Iterable[Dataset],
        collate_fn: Callable | None,
    ) -> list[DataLoader]:
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams["batch_size"],
                num_workers=self.hparams["num_workers"],
                collate_fn=collate_fn,
                pin_memory=self.hparams["pin_memory"],
                drop_last=False,
                shuffle=False,
            )
            for dataset in datasets
        ]
