"""CUDA Event timing shared by the playground and evaluation harness."""
import torch

TIMING_METHOD = "cuda_event_pipeline_v1"


def timed_inference(call):
    """Time a pipeline call on the current CUDA device, returning (result, seconds).

    Call inside the adapter scope: patch setup/teardown and image saving are
    excluded. Preserve the playground's record/synchronize ordering exactly.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda.synchronize()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) * 1e-3
