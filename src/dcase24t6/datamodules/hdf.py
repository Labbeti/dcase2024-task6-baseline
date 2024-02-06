#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Iterable, Literal

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchoutil.utils.data.collate import AdvancedCollateDict
from torchoutil.utils.data.dataset import EmptyDataset
from torchoutil.utils.hdf import HDFDataset

from dcase24t6.datamodules.aac import AACDatamodule
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

Stage = Literal["fit", "validate", "test", "predict"] | None
ALL_STAGES = ("fit", "validate", "test", "predict")


class HDFDatamodule(AACDatamodule):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        dataroot: str | Path,
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
        dataroot = Path(dataroot)
        if isinstance(train_hdfs, str):
            train_hdfs = (train_hdfs,)
        if isinstance(val_hdfs, str):
            val_hdfs = (val_hdfs,)
        if isinstance(test_hdfs, str):
            test_hdfs = (test_hdfs,)
        if isinstance(predict_hdfs, str):
            predict_hdfs = (predict_hdfs,)

        super().__init__()
        self.tokenizer = tokenizer
        self.dataroot = dataroot
        self.train_hdfs = train_hdfs
        self.val_hdfs = val_hdfs
        self.test_hdfs = test_hdfs
        self.predict_hdfs = predict_hdfs

        self.train_dataset = EmptyDataset()
        self.val_datasets = {}
        self.test_datasets = {}
        self.predict_datasets = {}
        self.collate_fn = None

        self.save_hyperparameters(ignore=["tokenizer", "pre_save_transform"])

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
        hdf_fnames = self.train_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            datasets[hdf_fname] = dataset

        dataset = ConcatDataset(datasets.values())
        self.train_dataset = dataset

        flat_references = [
            ref
            for dataset in datasets
            for refs in dataset[:, "captions"]
            for ref in refs
        ]
        self.tokenizer.train_from_iterator(flat_references)

        pad_values = {
            "audio": 0.0,
            "captions": self.tokenizer.pad_token_id,
            "mult_captions": self.tokenizer.pad_token_id,
        }
        self.collate_fn = AdvancedCollateDict(pad_values)

    def setup_val(self) -> None:
        hdf_fnames = self.val_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            datasets[hdf_fname] = dataset

        self.val_datasets = datasets

    def setup_test(self) -> None:
        hdf_fnames = self.test_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            datasets[hdf_fname] = dataset

        self.test_datasets = datasets

    def setup_predict(self) -> None:
        hdf_fnames = self.predict_hdfs

        datasets = {}
        for hdf_fname in hdf_fnames:
            hdf_fpath = self.dataroot.joinpath("HDF", hdf_fname)
            dataset = HDFDataset(hdf_fpath, return_shape_columns=True)
            datasets[hdf_fname] = dataset

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
