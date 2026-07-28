#!/usr/bin/bash
# CenterPoint PointPillars IGIE entry (single e2e SO).
#
# Usage:
#   ./run_igie.sh build        # export pt2 + build pp_e2e_fp16_ixinfer.so
#   ./run_igie.sh accuracy     # nuScenes mini val mAP/NDS
#   ./run_igie.sh performance  # single-frame IGIE FPS
#   ./run_igie.sh all          # build + accuracy
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

MODE="${1:-accuracy}"

# /opt/sw_home/enable is not nounset-safe; source before enabling -u.
if [[ -f /opt/sw_home/enable ]]; then
  # shellcheck disable=SC1091
  source /opt/sw_home/enable
fi
set -u

if [[ -z "${TVM_HOME:-}" ]]; then
  if [[ -d "${ROOT}/../igie" ]]; then
    TVM_HOME="$(cd "${ROOT}/../igie" && pwd)"
  else
    echo "error: set TVM_HOME to your igie root (e.g. export TVM_HOME=/path/to/igie)"
    exit 1
  fi
fi
export TVM_HOME

TORCH_LIB="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export PYTHONPATH="${ROOT}:${TVM_HOME}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${TVM_HOME}/build:/opt/sw_home/local/corex/lib64:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export NUSCENES_PATH="${NUSCENES_PATH:-${ROOT}/data/nuScenes}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CFG="configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini_igie_e2e.py"
CHECKPOINT="${CHECKPOINT:-${ROOT}/latest.pth}"
WORK_DIR="${WORK_DIR:-${ROOT}/work_dirs/igie_fp16_e2e}"
E2E_SO="${E2E_SO:-pp_e2e_fp16_ixinfer.so}"
E2E_PT2="${E2E_PT2:-${ROOT}/torch_model/pp_e2e.pt2}"
BENCH_WARMUP="${BENCH_WARMUP:-5}"
BENCH_RUNS="${BENCH_RUNS:-10}"

build_engine() {
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "error: checkpoint not found: ${CHECKPOINT}"
    exit 1
  fi
  if [[ ! -f "${E2E_PT2}" ]]; then
    python3 tools/export_e2e_pt2.py \
      --checkpoint "${CHECKPOINT}" \
      --pt2-path "${E2E_PT2}"
  fi
  python3 tools/build_e2e_engine.py \
    --pt2-path "${E2E_PT2}" \
    --engine-path "${ROOT}/${E2E_SO}" \
    --network-name e2e \
    --precision fp16 \
    --layout NHWC \
    --force
}

run_accuracy() {
  if [[ ! -f "${ROOT}/${E2E_SO}" ]]; then
    echo "${E2E_SO} not found; run: $0 build"
    exit 1
  fi
  python3 tools/igie_test.py \
    "${CFG}" \
    --work_dir "${WORK_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --gpus 1
}

run_performance() {
  if [[ ! -f "${ROOT}/${E2E_SO}" ]]; then
    echo "${E2E_SO} not found; run: $0 build"
    exit 1
  fi
  python3 tools/bench_e2e_fps.py \
    --checkpoint "${CHECKPOINT}" \
    --e2e-so "${E2E_SO}" \
    --warmup "${BENCH_WARMUP}" \
    --runs "${BENCH_RUNS}"
}

case "${MODE}" in
  build)
    build_engine
    ;;
  accuracy|infer)
    run_accuracy
    ;;
  performance|bench)
    run_performance
    ;;
  all)
    build_engine
    run_accuracy
    ;;
  *)
    echo "usage: $0 [build|accuracy|performance|all]"
    exit 1
    ;;
esac
