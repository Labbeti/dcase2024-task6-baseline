#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dcase24t6.datamodules.aac import AACDatamodule
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


class HDFDatamodule(AACDatamodule):
    def __init__(
        self,
        tokenizer: AACTokenizer,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def setup_train(self) -> None:
        pass

    def setup_val(self) -> None:
        pass

    def setup_test(self) -> None:
        pass

    def setup_predict(self) -> None:
        pass

    def teardown_train(self) -> None:
        pass

    def teardown_val(self) -> None:
        pass

    def teardown_test(self) -> None:
        pass

    def teardown_predict(self) -> None:
        pass
