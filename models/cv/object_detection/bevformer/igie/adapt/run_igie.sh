#!/usr/bin/bash
# BEVFormer three-stage FP16 IGIE entry.
#
# Usage:
#   ./run_igie.sh build         # export pt2 + build .so + numeric compare
#   ./run_igie.sh accuracy      # nuScenes mini mAP/NDS (needs .so)
#   ./run_igie.sh performance   # single-frame IGIE latency
#   ./run_igie.sh all           # build + accuracy
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ -f /opt/sw_home/enable ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/sw_home/enable
  set -u
fi
if [[ -z "${TVM_HOME:-}" ]]; then
  echo "ERROR: set TVM_HOME to your igie checkout (contains python/ and build/libtvm.so)." >&2
  exit 1
fi
export PYTHONPATH="${ROOT}:${TVM_HOME}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${TVM_HOME}/build:/opt/sw_home/local/corex/lib64:${LD_LIBRARY_PATH:-}"

MODE="${1:-accuracy}"

case "${MODE}" in
  build|e2e)
    bash tools/igie/run_igie_stacked_fp16.sh
    ;;
  accuracy|nuscenes|infer)
    SKIP_EXPORT=1 SKIP_BUILD=1 SKIP_COMPARE=1 RUN_NUSCENES=1 \
      bash tools/igie/run_igie_stacked_fp16.sh
    ;;
  performance|bench)
    python3 tools/igie/bench_stacked_fp16_fps.py --skip-torch
    ;;
  all)
    RUN_NUSCENES=1 bash tools/igie/run_igie_stacked_fp16.sh
    ;;
  *)
    echo "usage: $0 [build|accuracy|performance|all]"
    exit 1
    ;;
esac
