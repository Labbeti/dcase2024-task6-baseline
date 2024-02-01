#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal

from aac_datasets import Clotho
from torch.utils.data.dataloader import DataLoader
from torchoutil.utils.data.dataset import EmptyDataset

from dcase24t6.datamodules.aac import AACDatamodule
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


class HDFDatamodule(AACDatamodule):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        # Dataset args
        root: str,
        # DataLoader args
        batch_size: int = 32,
        num_workers: int | Literal["auto"] = "auto",
        pin_memory: bool = True,
        train_drop_last: bool = False,
        # Other args
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.root = root

        self.train_dataset = EmptyDataset()
        self.val_datasets = {}
        self.test_datasets = {}
        self.predict_datasets = {}
        self.collate_fn = None

        self.save_hyperparameters(ignore=["tokenizer", "root"])

    def prepare_data(self) -> None:
        datasets = {
            subset: Clotho(
                root=self.root,
                subset=subset,
                download=True,
                verbose=self.hparams["verbose"],
            )
            for subset in ("dev", "val", "eval", "dcase_aac_test", "dcase_aac_analysis")
        }
        for dataset in datasets.values():
            raise NotImplementedError

    def setup_train(self) -> None:
        raise NotImplementedError

    def setup_val(self) -> None:
        raise NotImplementedError

    def setup_test(self) -> None:
        raise NotImplementedError

    def setup_predict(self) -> None:
        raise NotImplementedError

    def teardown_train(self) -> None:
        self.train_dataset = EmptyDataset()

    def teardown_val(self) -> None:
        self.val_datasets = {}

    def teardown_test(self) -> None:
        self.test_datasets = {}

    def teardown_predict(self) -> None:
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
            )
            for dataset in datasets
        ]
