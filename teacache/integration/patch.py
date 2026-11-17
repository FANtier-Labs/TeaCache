"""Reversible, instance-local forward overrides."""
from contextlib import contextmanager


@contextmanager
def patch_forward(module, forward):
    had_override = 'forward' in vars(module)
    original = vars(module).get('forward')
    module.forward = forward
    try:
        yield
    finally:
        if had_override:
            module.forward = original
        else:
            del module.forward
