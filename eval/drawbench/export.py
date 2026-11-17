import argparse
import csv
import json
import math
from pathlib import Path
from .common import records, save, valid_seconds

FIELDS = (
    "lpips",
    "psnr",
    "ssim",
    "baseline_clip_score",
    "teacache_clip_score",
    "baseline_seconds",
    "teacache_seconds",
    "speedup",
)


def summarize(rows):
    if not rows:
        raise ValueError("no metric results")
    result = {}
    for key in FIELDS:
        values = [float(row[key]) for row in rows]
        if any(math.isnan(v) or v == -math.inf for v in values):
            raise ValueError("invalid score")
        if key != "psnr" and any(not math.isfinite(v) for v in values):
            raise ValueError("only PSNR may be infinite")
        result[key] = "inf" if math.inf in values else sum(values) / len(values)
    base_seconds = sum(valid_seconds(row["baseline_seconds"]) for row in rows)
    tea_seconds = sum(valid_seconds(row["teacache_seconds"]) for row in rows)
    result["speedup"] = base_seconds / tea_seconds
    result["baseline_total_seconds"] = base_seconds
    result["teacache_total_seconds"] = tea_seconds
    result["psnr_infinite_samples"] = sum(
        float(row["psnr"]) == math.inf for row in rows
    )
    finite = [float(r["psnr"]) for r in rows if math.isfinite(float(r["psnr"]))]
    result["psnr_finite_mean"] = sum(finite) / len(finite) if finite else None
    for key in ("clip_score",):
        result[key + "_delta"] = result["teacache_" + key] - result["baseline_" + key]
    return result


def build_report(root):
    root = Path(root)
    configs = [json.loads(p.read_text()) for p in root.glob("config.rank*.json")]
    if not configs or any(c != configs[0] for c in configs):
        raise ValueError("missing or incompatible metric configurations")
    config = json.loads((root / "config.json").read_text())
    if config != configs[0]:
        raise ValueError("incompatible metric configurations")
    rows = records(root)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(range(configs[0]["expected"])):
        raise ValueError("incomplete or duplicate metric results")
    categories = sorted({r["metadata"].get("category", "uncategorized") for r in rows})
    return dict(
        config=configs[0],
        samples=len(rows),
        overall=summarize(rows),
        categories={
            c: dict(
                samples=sum(
                    r["metadata"].get("category", "uncategorized") == c for r in rows
                ),
                **summarize(
                    [
                        r
                        for r in rows
                        if r["metadata"].get("category", "uncategorized") == c
                    ]
                ),
            )
            for c in categories
        },
    )


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(scores, output):
    report = build_report(scores)
    rows = sorted(records(scores), key=lambda row: row["sample_id"])
    samples = []
    for row in rows:
        samples.append(
            dict(
                sample_id=row["sample_id"],
                prompt=row["metadata"]["prompt"],
                category=row["metadata"].get("category", "uncategorized"),
                seed=row["seed"],
                **{key: row[key] for key in FIELDS},
                clip_score_delta=row["teacache_clip_score"]
                - row["baseline_clip_score"],
            )
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "summary.json", report)
    write_csv(
        output / "samples.csv",
        ["sample_id", "prompt", "category", "seed", *FIELDS, "clip_score_delta"],
        samples,
    )
    summaries = [
        dict(
            scope="overall", category="", samples=report["samples"], **report["overall"]
        )
    ]
    summaries.extend(
        dict(scope="category", category=category, **values)
        for category, values in report["categories"].items()
    )
    write_csv(output / "summary.csv", list(summaries[0]), summaries)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument(
        "--output", required=True, help="Directory for summary.json and CSV exports"
    )
    a = p.parse_args()
    export_results(a.scores, a.output)


if __name__ == "__main__":
    main()
