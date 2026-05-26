#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."   # project root
jupyter nbconvert --to notebook --execute sinkhorn_hti.ipynb \
    --inplace --ExecutePreprocessor.timeout=600
