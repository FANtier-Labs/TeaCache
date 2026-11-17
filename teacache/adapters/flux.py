"""FLUX.1-dev adapter for Diffusers 0.43.0's split text/image block API."""
from contextlib import ExitStack, contextmanager
from functools import wraps
from inspect import signature
import torch
from .base import BaseAdapter
from ..integration.patch import patch_forward


class FluxAdapter(BaseAdapter):
    @contextmanager
    def patch(self, pipeline, manager):
        transformer = pipeline.transformer
        if transformer.training or getattr(transformer, 'is_cache_enabled', False):
            raise ValueError('requires eval mode and disabled native cache')
        if getattr(transformer, '_teacache_active', False) or hasattr(transformer, '_orig_mod'):
            raise ValueError('nested patches and compiled transformers are unsupported')
        if getattr(manager, '_active', False):
            raise ValueError('manager already in use')
        blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
        if not transformer.transformer_blocks or not transformer.single_transformer_blocks:
            raise ValueError('requires both FLUX block stacks')
        original = transformer.forward
        sig = signature(original)
        state = {'step': -1, 'timestep': None}

        @wraps(original)
        def forward(*args, **kwargs):
            if torch.is_grad_enabled():
                raise ValueError('TeaCache requires torch.no_grad() or inference_mode()')
            bound = sig.bind(*args, **kwargs)
            values = bound.arguments
            if any(values.get(k) is not None for k in ('controlnet_block_samples', 'controlnet_single_block_samples')):
                raise ValueError('ControlNet is unsupported')
            attention = values.get('joint_attention_kwargs') or {}
            if any(k in attention for k in ('ip_adapter_image_embeds', 'ip_hidden_states')):
                raise ValueError('IP-Adapter is unsupported')
            timestep = pipeline.current_timestep
            if timestep is None:
                raise RuntimeError('requires an active FluxPipeline denoising loop')
            if timestep is state['timestep']:
                raise ValueError('true CFG / repeated transformer calls are unsupported')
            state['timestep'] = timestep
            state['step'] += 1
            state['num_steps'] = pipeline.num_timesteps
            return original(*args, **kwargs)

        manager.reset()
        manager._active = transformer._teacache_active = True
        try:
            with ExitStack() as stack:
                stack.enter_context(patch_forward(transformer, forward))
                for index, block in enumerate(blocks):
                    original_block = block.forward
                    @wraps(original_block)
                    def wrapped(hidden_states, encoder_hidden_states, temb, *args,
                                _index=index, _original=original_block, **kwargs):
                        if _index == 0:
                            modulated = blocks[0].norm1(hidden_states, emb=temb)[0]
                            state['compute'] = manager.should_compute(
                                modulated, hidden_states, state['step'], state['num_steps'])
                            state['input'] = hidden_states.detach().clone() if state['compute'] else None
                        if state['compute']:
                            text, image = _original(hidden_states, encoder_hidden_states, temb, *args, **kwargs)
                            if _index == len(blocks) - 1:
                                manager.update(state['input'], image)
                                state['input'] = None
                            return text, image
                        if _index == len(blocks) - 1:
                            hidden_states = manager.reuse(hidden_states)
                        return encoder_hidden_states, hidden_states
                    stack.enter_context(patch_forward(block, wrapped))
                yield
        finally:
            del transformer._teacache_active
            manager._active = False
            manager.reset()
