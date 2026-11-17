"""Per-request detached cache state."""
from dataclasses import dataclass
import torch


def compatible(left, right):
    return left is not None and (left.shape, left.device, left.dtype) == (right.shape, right.device, right.dtype)


@dataclass
class TeaCacheState:
    previous_input: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    accumulated_distance: float = 0.0
