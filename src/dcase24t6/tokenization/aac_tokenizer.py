#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Sequence, TypeGuard

from tokenizers import Encoding, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Normalizer
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.normalizers import Strip, StripAccents
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import PostProcessor, TemplateProcessing
from tokenizers.trainers import Trainer, WordLevelTrainer


class AACTokenizer:
    """Wrapper of tokenizers.Tokenizer to facilitate AAC development."""

    def __init__(
        self,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        post_processor: PostProcessor | None = None,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ) -> None:
        special_tokens = (pad_token, bos_token, eos_token, unk_token)
        initial_vocab = dict(zip(special_tokens, range(len(special_tokens))))
        model = WordLevel(initial_vocab, unk_token)

        if normalizer is None:
            normalizer = AACTokenizer.default_normalizer()
        if pre_tokenizer is None:
            pre_tokenizer = AACTokenizer.default_pre_tokenizer()
        if post_processor is None:
            post_processor = AACTokenizer.default_post_processor(
                bos_token, initial_vocab[bos_token], eos_token, initial_vocab[eos_token]
            )

        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizer  # type: ignore
        tokenizer.pre_tokenizer = pre_tokenizer  # type: ignore
        tokenizer.post_processor = post_processor  # type: ignore
        tokenizer.enable_padding(
            direction="right", pad_id=initial_vocab[pad_token], pad_token=pad_token
        )

        super().__init__()
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token
        self._tokenizer = tokenizer

    @classmethod
    def default_normalizer(cls) -> Normalizer:
        normalizer = NormalizerSequence(  # type: ignore
            [
                Lowercase(),
                Strip(),
                StripAccents(),
                # Replace("  ", " "),
            ],
        )
        return normalizer

    @classmethod
    def default_pre_tokenizer(cls) -> PreTokenizer:
        pre_tokenizer = PreTokenizerSequence(
            [
                Whitespace(),
            ],
        )
        return pre_tokenizer

    @classmethod
    def default_post_processor(
        cls, bos_token: str, bos_token_id: int, eos_token: str, eos_token_id: int
    ) -> PostProcessor:
        return TemplateProcessing(
            single=f"{bos_token} $0 {eos_token}",
            pair=None,
            special_tokens=[(bos_token, bos_token_id), (eos_token, eos_token_id)],
        )

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def special_tokens(self) -> list[str]:
        return [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id(self.pad_token)

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id(self.eos_token)

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id(self.unk_token)

    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)

    def train_from_iterator(
        self,
        sequence: list[str],
        trainer: Trainer | None = None,
    ) -> None:
        if trainer is None:
            trainer = WordLevelTrainer(special_tokens=self.special_tokens)  # type: ignore
        self.tokenizer.train_from_iterator(sequence, trainer=trainer)

    def encode(self, sequence: str, disable_unk_token: bool = False) -> Encoding:
        if disable_unk_token:
            self.tokenizer.model.unk_token = ""
        encoding = self.tokenizer.encode(sequence)
        if disable_unk_token:
            self.tokenizer.model.unk_token = self.unk_token
        return encoding

    def encode_batch(
        self, sequence: list[str], disable_unk_token: bool = False
    ) -> list[Encoding]:
        if disable_unk_token:
            self.tokenizer.model.unk_token = ""
        encodings = self.tokenizer.encode_batch(sequence)
        if disable_unk_token:
            self.tokenizer.model.unk_token = self.unk_token
        return encodings

    def decode(
        self, sequence: Sequence[int] | Encoding, skip_special_tokens: bool = True
    ) -> str:
        if isinstance(sequence, Encoding):
            sequence = sequence.ids
        decoded = self.tokenizer.decode(
            sequence, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def decode_batch(
        self,
        sequence: Sequence[Sequence[int]] | Sequence[Encoding],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if is_list_encoding(sequence):
            sequence = [element.ids for element in sequence]
        decoded = self.tokenizer.decode_batch(
            sequence, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def is_list_encoding(sequence: Any) -> TypeGuard[Sequence[Encoding]]:
    return isinstance(sequence, Sequence) and all(
        isinstance(element, Encoding) for element in sequence
    )
