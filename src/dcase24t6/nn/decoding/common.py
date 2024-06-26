#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Mapping, Optional, Protocol

import torch
from nltk.corpus import stopwords
from torch import Tensor

pylog = logging.getLogger(__name__)


class AACDecoder(Protocol):
    """Protocol for aac decoders. Similar to `torch.nn.TransformerDecoder` inputs."""

    def __call__(
        self,
        *,
        frame_embs: Tensor,
        frame_embs_pad_mask: Optional[Tensor],
        frame_embs_attn_mask: Optional[Tensor],
        caps_in: Tensor,
        caps_in_pad_mask: Optional[Tensor],
        caps_in_attn_mask: Tensor,
    ) -> Tensor:
        """Decode audio embeddings + previous captions tokens to next token logits.

        :param frame_embs: (n_frames, bsize, d_model)
        :param frame_embs_pad_mask: (bsize, n_frames) or None
        :param frame_embs_attn_mask: (caps_in_len, n_frames) or None
        :param caps_in: (caps_in_len, bsize, d_model)
        :param caps_in_pad_mask: (caps_in_len, bsize) or None
        :param caps_in_attn_mask: (caps_in_len, caps_in_len)
        :returns: logits of shape (caps_in_len, bsize, vocab_size)
        """
        raise NotImplementedError("Protocol abstract method.")


def get_forbid_rep_mask_content_words(
    vocab_size: int,
    token_to_id: Mapping[str, int],
    device: str | torch.device | None,
    verbose: int = 0,
    lang: str = "english",
) -> Tensor:
    forbid_rep_mask = torch.ones((vocab_size,), dtype=torch.bool, device=device)
    stopwords_set = set(stopwords.words(lang))
    stopwords_in_vocab = {word for word in stopwords_set if word in token_to_id}

    for token in stopwords_in_vocab:
        id_ = token_to_id[token]
        forbid_rep_mask[id_] = False

    if verbose >= 2:
        pylog.debug(
            f"{len(stopwords_in_vocab)}/{len(stopwords_set)} stopwords found in vocab:"
        )
        pylog.debug(f"{stopwords_in_vocab}")

        stopwords_not_in_vocab = {
            word for word in stopwords_set if word not in token_to_id
        }
        pylog.debug(
            f"{len(stopwords_not_in_vocab)}/{len(stopwords_set)} stopwords NOT found in vocab:"
        )
        pylog.debug(f"{stopwords_not_in_vocab}")
        pylog.debug(f"Found {len(stopwords_in_vocab)}/{len(stopwords_set)} stopwords.")

    if verbose >= 1:
        pylog.info(
            f"Forbid repetition mask {forbid_rep_mask.sum().item()}/{vocab_size} tokens during testing."
        )

    return forbid_rep_mask
