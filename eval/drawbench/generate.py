import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline

from teacache import CacheManager, FluxAdapter, TeaCacheConfig
from teacache.timing import TIMING_METHOD, timed_inference

from .common import load, save, fingerprint, tasks, ensure_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(Path(__file__).with_name("data.jsonl")),
        help="JSONL with prompt and original metadata",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["baseline", "teacache"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rel-l1-thresh", type=float, default=0.6)
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed requests per worker")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--limit", type=int, help="First N prompts for smoke tests")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.samples is not None and args.samples < 1:
        parser.error("samples must be positive")
    if args.warmup_runs < 0 or min(args.steps, args.height, args.width) < 1:
        parser.error("warmup-runs must be nonnegative; steps and dimensions positive")
    TeaCacheConfig(args.rel_l1_thresh)
    return args


@torch.inference_mode()
def main():
    # Parse arguments and prepare this worker's samples.
    args = parse_args()
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = "cuda:" + os.getenv("LOCAL_RANK", "0")
    torch.cuda.set_device(device)
    prompts = load(args.data)
    if not prompts or any(not row.get("prompt") for row in prompts):
        raise ValueError("empty dataset or missing prompt")
    if args.limit is not None:
        prompts = prompts[: args.limit]
    count = 1 if args.samples is None else args.samples
    root = Path(args.output)
    config = dict(
        vars(args),
        data_sha256=fingerprint(args.data),
        count=count,
        expected=len(prompts) * count,
        dtype="bfloat16",
        timing_method=TIMING_METHOD,
        generator_device="cpu",
        world_size=world,
        gpu_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
        diffusers_version=__import__("diffusers").__version__,
    )
    config.pop("output")
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(root / "config.json", config)
    local = list(tasks(prompts, count, args.seed, rank, world))
    local = [task for task in local if not (
        (root / f"{task['prompt_id']:05d}/samples/{task['sample_index']:04d}.png").is_file()
        and (root / f"records/{task['sample_id']:08d}.json").is_file())]
    if not local:
        return

    # Load the model and configure TeaCache.
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
        device
    )
    pipe.transformer.eval()
    manager = CacheManager(TeaCacheConfig(args.rel_l1_thresh))

    def generate(task, timed=False):
        # Each image gets its own patch scope and fresh cache.
        adapter = nullcontext()
        if args.method == "teacache":
            adapter = FluxAdapter().patch(pipe, manager)
        with adapter:
            call = lambda: pipe(
                prompt=task["metadata"]["prompt"],
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator("cpu").manual_seed(task["seed"]),
            ).images[0]
            return timed_inference(call) if timed else call()

    # Warm up before timing the dataset generation.
    for _ in range(args.warmup_runs):
        generate(local[0])
    for task in local:
        index = task["sample_id"]
        image = root / f"{task['prompt_id']:05d}/samples/{task['sample_index']:04d}.png"
        record = root / f"records/{index:08d}.json"
        if image.is_file() and record.is_file():
            continue
        result, seconds = generate(task, timed=True)
        # Save the image and its generation record.
        image.parent.mkdir(parents=True, exist_ok=True)
        result.save(image)
        save(image.parent.parent / "metadata.jsonl", task["metadata"])
        save(record, dict(task, image=str(image.resolve()), seconds=seconds, rank=rank))


if __name__ == "__main__":
    main()
