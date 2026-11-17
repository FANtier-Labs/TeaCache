#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-black-forest-labs/FLUX.1-dev}
OUTPUT=${OUTPUT:-outputs/drawbench/flux}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
# Space-separated thresholds; use identical generation arguments for run_base.sh.
THRESHOLDS=${THRESHOLDS:-0.25 0.4 0.6 0.8}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi
LAUNCH=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node "$NPROC_PER_NODE" --module)
read -r -a VALUES <<< "$THRESHOLDS"
for threshold in "${VALUES[@]}"; do
    RUN="$OUTPUT/threshold_$threshold"
    "${LAUNCH[@]}" eval.drawbench.generate --model "$MODEL" "$@" \
        --method teacache --rel-l1-thresh "$threshold" --output "$RUN/images"
    "${LAUNCH[@]}" eval.drawbench.score --baseline "$OUTPUT/baseline" \
        --candidate "$RUN/images" --output "$RUN/scores"
    "$PYTHON" -m eval.drawbench.export --scores "$RUN/scores" --output "$RUN/report"
done
