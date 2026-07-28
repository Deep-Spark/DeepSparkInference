#!/usr/bin/bash
# Tiny Py3.10 / modern-torch compatibility fixes on upstream CenterPoint.
# Usage (from CenterPoint root, after rsync adapt/):
#   bash apply_compat.sh
set -euo pipefail

# collections.Iterable -> collections.abc.Iterable (Python 3.10+)
for f in det3d/solver/fastai_optim.py det3d/solver/optim.py; do
  if [[ -f "${f}" ]] && grep -q 'from collections import Iterable' "${f}"; then
    sed -i 's/from collections import Iterable, defaultdict/from collections.abc import Iterable\nfrom collections import defaultdict/' "${f}"
  fi
done

# torch>=2.6 defaults weights_only=True
if [[ -f det3d/torchie/trainer/checkpoint.py ]] \
  && grep -q 'torch.load(filename, map_location=map_location)' det3d/torchie/trainer/checkpoint.py \
  && ! grep -q 'weights_only=False' det3d/torchie/trainer/checkpoint.py; then
  sed -i 's/torch.load(filename, map_location=map_location)/torch.load(filename, weights_only=False, map_location=map_location)/' \
    det3d/torchie/trainer/checkpoint.py
fi

echo "compat patches applied"
