"""Paired DrawBench metrics, one independent worker per device."""

import argparse
import json
import os
from pathlib import Path
from .common import paired_runs, save, fingerprint, ensure_config

CLIP_MODEL = "ViT-B/32"


class Metrics:
    def __init__(self, device, clip_model=CLIP_MODEL):
        import clip
        import lpips

        self.device = device
        self.clip, self.preprocess = clip.load(clip_model, device=device)
        self.clip.eval()
        self.lpips = lpips.LPIPS(net="alex", version="0.1").to(device).eval()

    def compute(self, prompt, baseline, candidate):
        import torch
        import clip
        from torchvision.transforms.functional import pil_to_tensor
        from torchmetrics.functional.image import structural_similarity_index_measure

        if baseline.size != candidate.size:
            raise ValueError("paired image sizes differ")
        with torch.inference_mode():
            a, b = [
                pil_to_tensor(im).unsqueeze(0).to(self.device).float() / 255
                for im in (baseline, candidate)
            ]
            mse = (a - b).square().mean()
            psnr = float(10 * torch.log10(1 / mse))
            row = dict(
                psnr="inf" if psnr == float("inf") else psnr,
                lpips=float(self.lpips(a, b, normalize=True)),
                ssim=float(structural_similarity_index_measure(
                    a, b, data_range=1.0, gaussian_kernel=True, sigma=1.5,
                    kernel_size=11, k1=0.01, k2=0.03, reduction="elementwise_mean",
                )),
            )
            text = self.clip.encode_text(
                clip.tokenize([prompt], truncate=True).to(self.device)
            ).float()
            text = torch.nn.functional.normalize(text, dim=-1)
            for label, im in (("baseline", baseline), ("teacache", candidate)):
                feature = self.clip.encode_image(
                    self.preprocess(im).unsqueeze(0).to(self.device)
                ).float()
                feature = torch.nn.functional.normalize(feature, dim=-1)
                row[label + "_clip_score"] = float((feature * text).sum())
            return row


def score_pairs(pairs, output, metrics, rank=0, world=1):
    from PIL import Image

    for a, b in pairs[rank::world]:
        path = Path(output) / "records" / f"{a['sample_id']:08d}.json"
        identity = dict(
            sample_id=a["sample_id"],
            seed=a["seed"],
            metadata=a["metadata"],
            baseline_sha256=fingerprint(a["image"]),
            candidate_sha256=fingerprint(b["image"]),
            baseline_seconds=a["seconds"],
            teacache_seconds=b["seconds"],
            speedup=a["seconds"] / b["seconds"],
        )
        if path.exists():
            existing = json.loads(path.read_text())
            if any(existing.get(key) != value for key, value in identity.items()):
                raise ValueError("score input changed; use a new output directory")
            continue
        with Image.open(a["image"]) as ia, Image.open(b["image"]) as ib:
            scores = metrics.compute(
                a["metadata"]["prompt"], ia.convert("RGB"), ib.convert("RGB")
            )
        save(path, dict(identity, **scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", help="Defaults to local CUDA device")
    p.add_argument("--clip-model", default=CLIP_MODEL)
    a = p.parse_args()
    bc, tc, pairs = paired_runs(a.baseline, a.candidate)
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = a.device or "cuda:" + os.getenv("LOCAL_RANK", "0")
    config = dict(
        baseline=bc,
        candidate=tc,
        clip_model=a.clip_model,
        lpips_net="alex",
        lpips_version="0.1",
        psnr_data_range=1,
        ssim=dict(
            implementation="torchmetrics.functional.image.structural_similarity_index_measure",
            data_range=1.0, gaussian_kernel=True, sigma=1.5, kernel_size=11,
            k1=0.01, k2=0.03, reduction="elementwise_mean",
        ),
        clip_definition="raw_cosine",
        text_truncate=True,
        expected=len(pairs),
        speedup_definition="sum(baseline_seconds) / sum(teacache_seconds)",
    )
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(root / "config.json", config)
    # Each rank has its own config; report verifies all workers agree.
    config_path = root / f"config.rank{rank}.json"
    for previous in root.glob("config.rank*.json"):
        if json.loads(previous.read_text()) != config:
            raise ValueError("metric configuration changed")
    save(config_path, config)
    if not pairs[rank::world]:
        return
    score_pairs(pairs, root, Metrics(device, a.clip_model), rank, world)


if __name__ == "__main__":
    main()
