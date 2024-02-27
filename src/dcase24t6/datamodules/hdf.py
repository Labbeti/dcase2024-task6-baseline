#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchoutil.utils.data.collate import AdvancedCollateDict
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import EmptyDataset, TransformWrapper
from torchoutil.utils.hdf import HDFDataset

from dcase24t6.datamodules.aac import AACDatamodule
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

Stage = Literal["fit", "validate", "test", "predict"] | None
ALL_STAGES = ("fit", "validate", "test", "predict")


class HDFDatamodule(AACDatamodule):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        root: str | Path,
        train_hdfs: str | Iterable[str] = (),
        val_hdfs: str | Iterable[str] = (),
        test_hdfs: str | Iterable[str] = (),
        predict_hdfs: str | Iterable[str] = (),
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
            train_hdfs = (train_hdfs,)
        if isinstance(val_hdfs, str):
            val_hdfs = (val_hdfs,)
        if isinstance(test_hdfs, str):
            test_hdfs = (test_hdfs,)
        if isinstance(predict_hdfs, str):
            predict_hdfs = (predict_hdfs,)
        if num_workers == "auto":
            num_workers = get_auto_num_cpus()

        super().__init__()
        self.tokenizer = tokenizer
        self.root = root
        self.train_hdfs = train_hdfs
        self.val_hdfs = val_hdfs
        self.test_hdfs = test_hdfs
        self.predict_hdfs = predict_hdfs

        self.train_dataset = EmptyDataset()
        self.val_datasets = {}
        self.test_datasets = {}
        self.predict_datasets = {}

        self.train_collate_fn = None
        self.val_collate_fn = None
        self.test_collate_fn = None
        self.predict_collate_fn = None

        self.save_hyperparameters(ignore=["tokenizer", "pre_save_transform"])

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
        return item

    def val_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        refs = item.pop("captions")
        item["mult_references"] = refs
        captions = self.tokenizer.encode_batch(refs)
        captions = torch.as_tensor([cap.ids for cap in captions])
        item["mult_captions"] = captions
        return item

    def test_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return self.val_transform(item)

    def predict_transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return self.val_transform(item)

    def setup_train(self) -> None:
        hdf_fnames = self.train_hdfs

        datasets = {}
        flat_references = []
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.root.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)

            # note: get captions before transform
            flat_references += [ref for refs in dataset[:, "captions"] for ref in refs]

            dataset = TransformWrapper(dataset, self.train_transform)
            datasets[hdf_fname] = dataset

        dataset = ConcatDataset(datasets.values())
        dataset = TransformWrapper(dataset, self.train_transform)

        pad_values = {
            "frame_embs": 0.0,
            "captions": self.tokenizer.pad_token_id,
        }
        train_collate_fn = AdvancedCollateDict(pad_values)

        self.train_dataset = dataset
        self.train_collate_fn = train_collate_fn
        self.tokenizer.train_from_iterator(flat_references)

    def setup_val(self) -> None:
        hdf_fnames = self.val_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.root.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            dataset = TransformWrapper(dataset, self.val_transform)
            datasets[hdf_fname] = dataset

        pad_values = {
            "frame_embs": 0.0,
            "mult_captions": self.tokenizer.pad_token_id,
        }
        val_collate_fn = AdvancedCollateDict(pad_values)

        self.val_datasets = datasets
        self.val_collate_fn = val_collate_fn

    def setup_test(self) -> None:
        hdf_fnames = self.test_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.root.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            dataset = TransformWrapper(dataset, self.test_transform)
            datasets[hdf_fname] = dataset

        pad_values = {
            "frame_embs": 0.0,
            "mult_captions": self.tokenizer.pad_token_id,
        }
        test_collate_fn = AdvancedCollateDict(pad_values)

        self.test_datasets = datasets
        self.test_collate_fn = test_collate_fn

    def setup_predict(self) -> None:
        hdf_fnames = self.predict_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.root.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            dataset = TransformWrapper(dataset, self.predict_transform)
            datasets[hdf_fname] = dataset

        pad_values = {
            "frame_embs": 0.0,
            "mult_captions": self.tokenizer.pad_token_id,
        }
        predict_collate_fn = AdvancedCollateDict(pad_values)

        self.predict_datasets = datasets
        self.predict_collate_fn = predict_collate_fn

    def teardown_train(self) -> None:
        if isinstance(self.train_dataset, HDFDataset):
            self.train_dataset.close()
        self.train_dataset = EmptyDataset()
        self.train_collate_fn = None

    def teardown_val(self) -> None:
        for dataset in self.val_datasets.values():
            if isinstance(dataset, HDFDataset):
                dataset.close()
        self.val_datasets = {}
        self.val_collate_fn = None

    def teardown_test(self) -> None:
        for dataset in self.test_datasets.values():
            if isinstance(dataset, HDFDataset):
                dataset.close()
        self.test_datasets = {}
        self.test_collate_fn = None

    def teardown_predict(self) -> None:
        for dataset in self.predict_datasets.values():
            if isinstance(dataset, HDFDataset):
                dataset.close()
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
        datasets = self.val_datasets.values()
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams["batch_size"],
                num_workers=self.hparams["num_workers"],
                collate_fn=self.val_collate_fn,
                pin_memory=self.hparams["pin_memory"],
                drop_last=False,
                shuffle=False,
            )
            for dataset in datasets
        ]

    def test_dataloader(self) -> list[DataLoader]:
        datasets = self.test_datasets.values()
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams["batch_size"],
                num_workers=self.hparams["num_workers"],
                collate_fn=self.test_collate_fn,
                pin_memory=self.hparams["pin_memory"],
                drop_last=False,
                shuffle=False,
            )
            for dataset in datasets
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        datasets = self.predict_datasets.values()
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams["batch_size"],
                num_workers=self.hparams["num_workers"],
                collate_fn=self.predict_collate_fn,
                pin_memory=self.hparams["pin_memory"],
                drop_last=False,
                shuffle=False,
            )
            for dataset in datasets
        ]
