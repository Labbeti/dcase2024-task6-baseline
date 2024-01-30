#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable

import torch
from torch import Tensor
from torch.distributions.beta import Beta


def sample_lambda(
    alpha: float | Tensor,
    asymmetric: bool,
    size: Iterable[int] = (),
) -> Tensor:
    """
    :param alpha: alpha hp to control the Beta distribution.
        Values closes to 0 means distribution will peak up at 0 and 1, while values closes to 1 means sampling from an uniform distribution.
    :param asymmetric: If True, lbd value will always be in [0.5, 1], with values close to 1.
    :param size: The size of the sampled lambda(s) value(s). defaults to ().
    :returns: Sampled values of shape defined by size argument.
    """
    tensor_kwds: dict[str, Any] = dict(dtype=torch.get_default_dtype())
    size = torch.Size(size)

    if alpha == 0.0:
        if asymmetric:
            return torch.full(size, 1.0, **tensor_kwds)
        else:
            return torch.rand(size).ge(0.5).to(**tensor_kwds)

    alpha = torch.as_tensor(alpha, **tensor_kwds)
    beta = Beta(alpha, alpha)
    lbd = beta.sample(size)
    if asymmetric:
        lbd = torch.max(lbd, 1.0 - lbd)
    return lbd
