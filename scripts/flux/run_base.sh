#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-black-forest-labs/FLUX.1-dev}
OUTPUT=${OUTPUT:-outputs/drawbench/flux}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node "$NPROC_PER_NODE" --module eval.drawbench.generate \
    --model "$MODEL" "$@" --method baseline --output "$OUTPUT/baseline"
