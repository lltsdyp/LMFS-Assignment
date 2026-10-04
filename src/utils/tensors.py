"""Shared tensor accumulation dtype and input-device validation."""

import torch
from torch import Tensor


def compute_dtype(x: Tensor) -> torch.dtype:
    """Use float32 accumulation for half precision and retain float64."""
    return (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )


def check_same_device(*tensors: Tensor) -> None:
    """Require inputs on one device without selecting an execution backend."""
    if not tensors:
        raise ValueError("at least one tensor is required")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
