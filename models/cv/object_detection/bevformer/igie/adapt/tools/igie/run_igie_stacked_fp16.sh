#!/usr/bin/bash
# Three-stage fp16 IGIE pipeline: export pt2 -> build .so -> numeric compare -> optional nuScenes.
#
# Usage:
#   ./run_igie.sh all
#   ./run_igie.sh e2e
#   RUN_NUSCENES=1 ./run_igie.sh all
#   SKIP_EXPORT=1 SKIP_BUILD=1 ./run_igie.sh e2e
#   SKIP_EXPORT=1 SKIP_BUILD=1 SKIP_COMPARE=1 RUN_NUSCENES=1 ./run_igie.sh nuscenes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-projects/configs/bevformer/bevformer-base.py}"
CKPT="${CKPT:-./ckpts/bevformer-base.pth}"
SHAPE="${SHAPE:-1 6 3 928 1600}"
LAYOUT="${LAYOUT:-NHWC}"

BACKBONE_PT2="${BACKBONE_PT2:-bevformer_backbone.pt2}"
ENCODER_PT2="${ENCODER_PT2:-bevformer_encoder.pt2}"
DECODER_PT2="${DECODER_PT2:-bevformer_decoder.pt2}"

BACKBONE_SO="${BACKBONE_SO:-bevformer_backbone_fp16_conv_only_ixinfer_${LAYOUT}_mdcn.so}"
ENCODER_SO="${ENCODER_SO:-bevformer_encoder_fp16_ixinfer_${LAYOUT}_msdeform.so}"
DECODER_SO="${DECODER_SO:-bevformer_decoder_fp16_ixinfer_${LAYOUT}_msdeform.so}"

BUNDLE="${BUNDLE:-stacked_fp16_compare_bundle.npz}"
JSON_PREFIX="${JSON_PREFIX:-./test/igie_stacked_fp16}"
LOG="${LOG:-nuscenes_igie_stacked_fp16.log}"

SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"
RUN_NUSCENES="${RUN_NUSCENES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ -f /opt/sw_home/enable ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/sw_home/enable
  set -u
fi
if [[ -z "${TVM_HOME:-}" ]]; then
  echo "ERROR: set TVM_HOME to your igie checkout (compiled IGIE/TVM root)." >&2
  exit 1
fi
export PYTHONPATH="${TVM_HOME}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${TVM_HOME}/build:/opt/sw_home/local/corex/lib64:${LD_LIBRARY_PATH:-}"

read -r B N C H W <<< "${SHAPE}"

install_mmcv_export_patches() {
  # Stock mmcv cannot torch.export encoder (needs level_lengths / no int() on shapes).
  local patch_dir="${ROOT}/tools/igie/patches"
  local msda_dst mdcn_dst
  msda_dst="$(python3 -c "import mmcv.ops.multi_scale_deform_attn as m; print(m.__file__)")"
  mdcn_dst="$(python3 -c "import mmcv.ops.modulated_deform_conv as m; print(m.__file__)")"
  if [[ -f "${patch_dir}/multi_scale_deform_attn.py" ]]; then
    cp -f "${patch_dir}/multi_scale_deform_attn.py" "${msda_dst}"
    echo "installed export-friendly ms_deform -> ${msda_dst}"
  elif [[ -f mmcv-src/mmcv/ops/multi_scale_deform_attn.py ]]; then
    cp -f mmcv-src/mmcv/ops/multi_scale_deform_attn.py "${msda_dst}"
    echo "restored export-friendly ms_deform from mmcv-src -> ${msda_dst}"
  else
    python3 tools/igie/apply_mmcv_ms_deform_patch.py "${msda_dst}"
  fi
  if [[ -f "${patch_dir}/modulated_deform_conv.py" ]]; then
    cp -f "${patch_dir}/modulated_deform_conv.py" "${mdcn_dst}"
    echo "installed export-friendly mdcn -> ${mdcn_dst}"
  else
    python3 tools/igie/apply_mmcv_dcn_patch.py 2>/dev/null || true
  fi
}

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  echo "==> [1/3] export 3-stage pt2"
  install_mmcv_export_patches
  python3 tools/igie/export_bevformer_backbone_pt2.py \
    "${CONFIG}" "${CKPT}" --shape "${B}" "${N}" "${C}" "${H}" "${W}" \
    --output "${BACKBONE_PT2}"
  python3 tools/igie/export_encoder_pt2.py \
    "${CONFIG}" "${CKPT}" --image-shape "${B}" "${N}" "${C}" "${H}" "${W}" \
    --output "${ENCODER_PT2}"
  python3 tools/igie/export_decoder_pt2.py \
    "${CONFIG}" "${CKPT}" --batch-size "${B}" --output "${DECODER_PT2}"
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "==> [2/3] build fp16 engines"
  python3 tools/igie/build_engine.py \
    --model_path "${BACKBONE_PT2}" \
    --engine_path "${BACKBONE_SO}" \
    --precision fp16 --layout "${LAYOUT}" --force
  python3 tools/igie/build_engine.py \
    --model_path "${ENCODER_PT2}" \
    --engine_path "${ENCODER_SO}" \
    --precision fp16 --layout "${LAYOUT}" --force
  python3 tools/igie/build_engine.py \
    --model_path "${DECODER_PT2}" \
    --engine_path "${DECODER_SO}" \
    --precision fp16 --layout "${LAYOUT}" --force
fi

if [[ "${SKIP_COMPARE}" != "1" ]]; then
  echo "==> [3/3] numeric compare (PT2 fp32 reference vs fp16 stack)"
  python3 tools/igie/compare_stacked_fp16.py \
    --backbone-pt2 "${BACKBONE_PT2}" \
    --encoder-pt2 "${ENCODER_PT2}" \
    --decoder-pt2 "${DECODER_PT2}" \
    --bundle "${BUNDLE}" \
    --image-shape "${B}" "${N}" "${C}" "${H}" "${W}" \
    --backbone-so "${BACKBONE_SO}" \
    --encoder-so "${ENCODER_SO}" \
    --decoder-so "${DECODER_SO}" \
    --fail-on-mismatch
fi

if [[ "${RUN_NUSCENES}" == "1" ]]; then
  echo "==> nuScenes stacked fp16 eval"
  NUS_ARGS=(
    "${CONFIG}" "${CKPT}"
    --backbone-so "${BACKBONE_SO}"
    --encoder-so "${ENCODER_SO}"
    --decoder-so "${DECODER_SO}"
    --jsonfile-prefix "${JSON_PREFIX}"
    --show-progress
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    NUS_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  python3 tools/igie/inference_stacked_nuscenes.py "${NUS_ARGS[@]}" 2>&1 | tee "${LOG}"
fi

echo "stacked fp16 IGIE done."
