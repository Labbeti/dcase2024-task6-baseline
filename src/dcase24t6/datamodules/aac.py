#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Literal

from lightning import LightningDataModule

Stage = Literal["fit", "validate", "test"] | None


class AACDatamodule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

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

    @abstractmethod
    def setup_train(self) -> None:
        pass

    @abstractmethod
    def setup_val(self) -> None:
        pass

    @abstractmethod
    def setup_test(self) -> None:
        pass

    @abstractmethod
    def setup_predict(self) -> None:
        pass

    @abstractmethod
    def teardown_train(self) -> None:
        pass

    @abstractmethod
    def teardown_val(self) -> None:
        pass

    @abstractmethod
    def teardown_test(self) -> None:
        pass

    @abstractmethod
    def teardown_predict(self) -> None:
        pass
