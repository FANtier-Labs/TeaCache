from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TeaCacheConfig:
    """Configuration of TeaCache.

    - `rel_l1_thresh`: Zero disables skipping; larger thresholds permit more feature reuse.
    """

    rel_l1_thresh: float = 0.6

    def __post_init__(self):
        if not math.isfinite(self.rel_l1_thresh) or self.rel_l1_thresh < 0:
            raise ValueError("rel_l1_thresh must be finite and nonnegative")
