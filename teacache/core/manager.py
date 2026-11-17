import math
import torch
from ..config import TeaCacheConfig
from .cache import TeaCacheState, compatible
from .scheduler import rescale


class CacheManager:
    """Own one request's cache. Do not share across concurrent requests."""
    def __init__(self, config: TeaCacheConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.state = TeaCacheState()
        self.last_step = -1

    def should_compute(self, modulated, hidden, step, num_steps):
        if not 0 <= step < num_steps or step <= self.last_step:
            raise ValueError("steps must increase within one generation request")
        state = self.state
        compute = (step == 0 or step == num_steps - 1 or self.config.rel_l1_thresh == 0
                   or not compatible(state.previous_input, modulated)
                   or not compatible(state.residual, hidden))
        if not compute:
            previous = state.previous_input.float()
            denominator = previous.abs().mean().item()
            distance = ((modulated.float() - previous).abs().mean().item() / denominator
                        if denominator > 0 else float('nan'))
            mapped = rescale(distance)
            state.accumulated_distance += mapped
            compute = (not math.isfinite(state.accumulated_distance)
                       or state.accumulated_distance >= self.config.rel_l1_thresh)
        if compute:
            state.accumulated_distance = 0.0
        state.previous_input = modulated.detach().clone()
        self.last_step = step
        return compute

    def update(self, before, after):
        residual = (after - before).detach()
        self.state.residual = residual if torch.isfinite(residual).all().item() else None

    def reuse(self, hidden):
        if not compatible(self.state.residual, hidden):
            raise RuntimeError("no compatible residual")
        return hidden + self.state.residual
