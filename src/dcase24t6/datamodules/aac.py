#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Literal

from lightning import LightningDataModule

Stage = Literal["fit", "validate", "test"] | None


class AACDatamodule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

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
