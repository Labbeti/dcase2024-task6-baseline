#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, ClassVar, Sequence, TypeGuard

from tokenizers import Encoding, Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Normalizer, Replace
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.normalizers import Strip, StripAccents
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import PostProcessor
from tokenizers.processors import Sequence as ProcessorSequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import Trainer, WordLevelTrainer


class AACTokenizer:
    """Wrapper of tokenizers.Tokenizer."""

    VERSION: ClassVar[int] = 1

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        version: int | None = None,
    ) -> None:
        if tokenizer is None:
            special_tokens = (pad_token, bos_token, eos_token, unk_token)
            initial_vocab = dict(zip(special_tokens, range(len(special_tokens))))
            model = WordLevel(initial_vocab, unk_token)

            normalizer = AACTokenizer.default_normalizer()
            pre_tokenizer = AACTokenizer.default_pre_tokenizer()
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

        if version is None:
            version = AACTokenizer.VERSION

        super().__init__()
        self._tokenizer = tokenizer
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token
        self._version = version

    @classmethod
    def default_normalizer(cls) -> Normalizer:
        normalizer = NormalizerSequence(  # type: ignore
            [
                Lowercase(),
                Strip(),
                StripAccents(),
                Replace(r"“", '"'),
                Replace(r"”", '"'),
                Replace(r"`", "'"),
                Replace(r"’", "'"),
                Replace(r";", ","),
                Replace(r"…", "..."),
                Replace(Regex(r"\s*-\s*"), "-"),
                # Replace all punctuation and weird characters except comma
                Replace(Regex(r"[.!?;:\"“”’`\(\)\{\}\[\]\*\×\-#/+_~ʘ\\/]"), " "),
                Replace(Regex(r"\s+"), " "),
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
        cls,
        bos_token: str,
        bos_token_id: int,
        eos_token: str,
        eos_token_id: int,
    ) -> PostProcessor:
        return ProcessorSequence(
            [
                TemplateProcessing(
                    single=f"{bos_token} $0 {eos_token}",
                    pair=None,
                    special_tokens=[
                        (bos_token, bos_token_id),
                        (eos_token, eos_token_id),
                    ],
                ),
            ],
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

    def get_token_to_id(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def get_id_to_token(self) -> dict[int, str]:
        return {id_: token for token, id_ in self.get_token_to_id().items()}

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
        self,
        sequence: list[str],
        disable_unk_token: bool = False,
    ) -> list[Encoding]:
        if disable_unk_token:
            self.tokenizer.model.unk_token = ""
        encodings = self.tokenizer.encode_batch(sequence)
        if disable_unk_token:
            self.tokenizer.model.unk_token = self.unk_token
        return encodings

    def decode(
        self,
        sequence: Sequence[int] | Encoding,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(sequence, Encoding):
            sequence = sequence.ids

        decoded = self.tokenizer.decode(
            sequence, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def decode_batch(
        self,
        sequences: Sequence[Sequence[int]] | Sequence[Encoding],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if is_list_encoding(sequences):
            sequences = [element.ids for element in sequences]
        decoded = self.tokenizer.decode_batch(
            sequences, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def to_str(self, pretty: bool = False) -> str:
        added_data = {
            "pad_token": self._pad_token,
            "bos_token": self._bos_token,
            "eos_token": self._eos_token,
            "unk_token": self._unk_token,
            "version": self._version,
        }
        tokenizer_data = json.loads(self.tokenizer.to_str())
        tokenizer_data |= added_data
        indent = 2 if pretty else None
        content = json.dumps(tokenizer_data, indent=indent)
        return content

    @classmethod
    def from_str(cls, content: str) -> "AACTokenizer":
        parsed = json.loads(content)
        added_data = {
            name: parsed.pop(name)
            for name in ("pad_token", "bos_token", "eos_token", "unk_token", "version")
        }
        tokenizer = Tokenizer.from_str(json.dumps(parsed))
        aac_tokenizer = AACTokenizer(tokenizer, **added_data)
        return aac_tokenizer

    def save(self, path: str | Path, pretty: bool = True) -> None:
        """Save tokenizer to JSON file."""
        path = Path(path).resolve()
        content = self.to_str(pretty=pretty)
        path.write_text(content)

    @classmethod
    def from_file(cls, path: str | Path) -> "AACTokenizer":
        """Load tokenizer from JSON file."""
        path = Path(path).resolve()
        content = path.read_text()
        aac_tokenizer = cls.from_str(content)
        return aac_tokenizer


def is_list_encoding(sequence: Any) -> TypeGuard[Sequence[Encoding]]:
    return isinstance(sequence, Sequence) and all(
        isinstance(element, Encoding) for element in sequence
    )
