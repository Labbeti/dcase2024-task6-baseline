#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Callable, Literal

from aac_datasets import Clotho
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchoutil.utils.data.dataset import EmptyDataset
from torchoutil.utils.data.hdf import HDFDataset, pack_to_hdf

from dcase24t6.datamodules.aac import AACDatamodule, Stage
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


class HDFDatamodule(AACDatamodule):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        pre_save_transform: Callable | None,
        dataroot: str | Path,
        hdf_name_pattern: str,
        # DataLoader args
        batch_size: int = 32,
        num_workers: int | Literal["auto"] = "auto",
        pin_memory: bool = True,
        train_drop_last: bool = False,
        # Other args
        verbose: int = 0,
    ) -> None:
        dataroot = Path(dataroot)

        super().__init__()
        self.tokenizer = tokenizer
        self.pre_save_transform = pre_save_transform

        self.dataroot = dataroot
        self.hdf_name_pattern = hdf_name_pattern

        self.train_dataset = EmptyDataset()
        self.val_datasets = {}
        self.test_datasets = {}
        self.predict_datasets = {}
        self.collate_fn = None

        self.save_hyperparameters(ignore=["tokenizer", "pre_save_transform"])

    def prepare_data(self) -> None:
        subsets = ("dev", "val", "eval", "dcase_aac_test", "dcase_aac_analysis")
        datasets = {
            subset: Clotho(
                root=self.dataroot,
                subset=subset,
                download=True,
                verbose=self.hparams["verbose"],
            )
            for subset in subsets
        }

        hdf_root = self.dataroot.joinpath("HDF")
        os.makedirs(hdf_root, exist_ok=True)

        hdf_datasets = {}
        for subset, dataset in datasets.items():
            hdf_fname = self.hdf_name_pattern.format(subset=subset)
            hdf_fpath = hdf_root.joinpath(hdf_fname)
            if hdf_fpath.exists():
                continue

            hdf_dataset = pack_to_hdf(
                dataset,
                hdf_fpath,
                self.pre_save_transform,
                batch_size=self.hparams["batch_size"],
                num_workers=self.hparams["num_workers"],
                verbose=self.hparams["verbose"],
            )
            hdf_datasets[subset] = hdf_dataset

    def setup(self, stage: Stage = None) -> None:
        match stage:
            case "fit":
                self.setup_train()
                self.setup_val()
            case "validate":
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
                self.teardown_val()
            case "test":
                self.teardown_test()
            case "predict":
                self.teardown_predict()
            case None:
                self.teardown_train()
                self.teardown_val()
                self.teardown_test()
                self.teardown_predict()

    def setup_train(self) -> None:
        subsets = ["dev"]

        datasets = {}
        for subset in subsets:
            hdf_fname = self.hdf_name_pattern.format(subset=subset)
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath)
            datasets[subset] = dataset

        dataset = ConcatDataset(datasets.values())
        self.train_dataset = dataset

        flat_references = [
            ref
            for dataset in datasets
            for refs in dataset[:, "captions"]
            for ref in refs
        ]
        self.tokenizer.train_from_iterator(flat_references)

    def setup_val(self) -> None:
        subsets = ["val"]

        datasets = {}
        for subset in subsets:
            hdf_fname = self.hdf_name_pattern.format(subset=subset)
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath)
            datasets[subset] = dataset

        self.val_datasets = datasets

    def setup_test(self) -> None:
        subsets = ["val", "eval"]

        datasets = {}
        for subset in subsets:
            hdf_fname = self.hdf_name_pattern.format(subset=subset)
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath)
            datasets[subset] = dataset

        self.test_datasets = datasets

    def setup_predict(self) -> None:
        subsets = ["dcase_aac_test", "dcase_aac_analysis"]

        datasets = {}
        for subset in subsets:
            hdf_fname = self.hdf_name_pattern.format(subset=subset)
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath)
            datasets[subset] = dataset

        self.predict_datasets = datasets

    def teardown_train(self) -> None:
        if isinstance(self.train_dataset, HDFDataset):
            self.train_dataset.close()
        self.train_dataset = EmptyDataset()

    def teardown_val(self) -> None:
        for dataset in self.val_datasets.values():
            if isinstance(dataset, HDFDataset):
                dataset.close()
        self.val_datasets = {}

    def teardown_test(self) -> None:
        for dataset in self.test_datasets.values():
            if isinstance(dataset, HDFDataset):
                dataset.close()
        self.test_datasets = {}

    def teardown_predict(self) -> None:
        for dataset in self.predict_datasets.values():
            if isinstance(dataset, HDFDataset):
                dataset.close()
        self.predict_datasets = {}

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collate_fn,
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
                collate_fn=self.collate_fn,
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
                collate_fn=self.collate_fn,
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
                collate_fn=self.collate_fn,
                pin_memory=self.hparams["pin_memory"],
                drop_last=False,
                shuffle=False,
            )
            for dataset in datasets
        ]
