#!/usr/bin/bash
# Pre-flight checks for BEVFormer IGIE three-stage fp16 reproduction.
# Usage (from BEVFormer repo root):
#   export TVM_HOME=/path/to/igie
#   bash tools/igie/check_repro_prereqs.sh
#
# Exit 0 if all required checks pass; non-zero otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

fail=0
warn=0

ok()   { echo "  [OK] $*"; }
bad()  { echo "  [FAIL] $*"; fail=1; }
note() { echo "  [WARN] $*"; warn=1; }

echo "==> BEVFormer repo"
ok "cwd=${ROOT}"

echo "==> Iluvatar runtime"
if [[ -f /opt/sw_home/enable ]]; then
  # shellcheck disable=SC1091
  set +u; source /opt/sw_home/enable; set -u
  ok "sourced /opt/sw_home/enable"
else
  bad "/opt/sw_home/enable missing (need Iluvatar MR100 container)"
fi

echo "==> TVM_HOME / IGIE"
if [[ -z "${TVM_HOME:-}" ]]; then
  bad "TVM_HOME unset. Example: export TVM_HOME=/path/to/igie"
else
  ok "TVM_HOME=${TVM_HOME}"
  [[ -f "${TVM_HOME}/build/libtvm.so" ]] && ok "libtvm.so present" \
    || bad "missing ${TVM_HOME}/build/libtvm.so (run igie build)"
  [[ -d "${TVM_HOME}/python/tvm" ]] && ok "python/tvm present" \
    || bad "missing ${TVM_HOME}/python/tvm"

  # Required passes for the published accuracy/latency numbers
  LF16="${TVM_HOME}/python/tvm/relax/transform/iluvatar/legalize_float16.py"
  DEFORM="${TVM_HOME}/python/tvm/relax/transform/iluvatar/deform_conv_rewriters.py"
  PIPE="${TVM_HOME}/python/tvm/relax/transform/iluvatar/deform_conv_pipeline.py"
  if [[ -f "${LF16}" ]]; then
    grep -q 'MHANoMask3DFMHARewriter' "${LF16}" \
      && ok "pass MHANoMask3DFMHARewriter (decoder FMHA)" \
      || bad "missing MHANoMask3DFMHARewriter in ${LF16}"
    grep -q 'LegalizeIxinferGemmOnlyFP16' "${LF16}" \
      && ok "pass LegalizeIxinferGemmOnlyFP16 (enc/dec GEMM fp16)" \
      || bad "missing LegalizeIxinferGemmOnlyFP16 in ${LF16}"
  else
    bad "missing ${LF16}"
  fi
  if [[ -f "${DEFORM}" ]]; then
    grep -q 'LegalizeFoldDCNFrozenBN\|FoldDCNFrozenBN' "${DEFORM}" \
      && ok "pass FoldDCNFrozenBN (backbone DCN fp16)" \
      || note "FoldDCNFrozenBN not found — backbone may be slower than ~313ms"
    grep -q 'MSDeformAttnRewriter\|ModulatedDeformConvRewriter' "${DEFORM}" \
      && ok "deform rewriters present" \
      || bad "missing MSDeform/MDCN rewriters in ${DEFORM}"
    if [[ -f "${PIPE}" ]]; then
      grep -q 'LegalizeFuseDeformExportSubgraphs' "${PIPE}" \
        && ok "pipeline LegalizeFuseDeformExportSubgraphs" \
        || bad "missing LegalizeFuseDeformExportSubgraphs in ${PIPE}"
      grep -q 'LegalizeModulatedDeformConvFp16Graph' "${PIPE}" \
        && ok "pipeline LegalizeModulatedDeformConvFp16Graph (DCN-gated fp16)" \
        || bad "missing LegalizeModulatedDeformConvFp16Graph in ${PIPE}"
    else
      bad "missing ${PIPE}"
    fi
  else
    bad "missing ${DEFORM}"
  fi
fi

echo "==> Python imports"
export PYTHONPATH="${ROOT}:${TVM_HOME:-}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${TVM_HOME:-}/build:/opt/sw_home/local/corex/lib64:${LD_LIBRARY_PATH:-}"
if python3 -c "import torch; print(torch.__version__)" >/tmp/bev_chk_torch.txt 2>&1; then
  ok "torch $(cat /tmp/bev_chk_torch.txt)"
else
  bad "import torch failed — see Python env section in README_CN.md"
fi
if python3 -c "import tvm; from tvm import relax; print(tvm.__file__)" >/tmp/bev_chk_tvm.txt 2>&1; then
  ok "tvm from $(cat /tmp/bev_chk_tvm.txt)"
else
  bad "import tvm failed — check TVM_HOME/PYTHONPATH"
fi

echo "==> Checkpoint / data"
[[ -f ckpts/bevformer-base.pth ]] && ok "ckpts/bevformer-base.pth" \
  || bad "missing ckpts/bevformer-base.pth"
[[ -f data/nuscenes/nuscenes_infos_temporal_val.pkl ]] && ok "val pkl" \
  || bad "missing data/nuscenes/nuscenes_infos_temporal_val.pkl (run create_data.py)"
[[ -d data/nuscenes/v1.0-mini || -d data/nuscenes/v1.0-trainval ]] \
  && ok "nuScenes meta dir present" \
  || note "nuScenes version dir not found under data/nuscenes/"

echo "==> Engines (optional for e2e; required for ./run_igie.sh nuscenes)"
for so in \
  bevformer_backbone_fp16_conv_only_ixinfer_NHWC_mdcn.so \
  bevformer_encoder_fp16_ixinfer_NHWC_msdeform.so \
  bevformer_decoder_fp16_ixinfer_NHWC_msdeform.so
do
  if [[ -f "${so}" ]]; then
    ok "${so}"
  else
    note "missing ${so} — run ./run_igie.sh e2e to build"
  fi
done

if [[ -f bevformer_decoder_fp16_ixinfer_NHWC_msdeform.so ]]; then
  n_fmha=$(strings bevformer_decoder_fp16_ixinfer_NHWC_msdeform.so | grep -c FMHA || true)
  if [[ "${n_fmha}" -gt 0 ]]; then
    ok "decoder SO embeds FMHA (count=${n_fmha})"
  else
    note "decoder SO has no FMHA string — rebuild with current IGIE (MHANoMask3DFMHARewriter)"
  fi
fi
if [[ -f bevformer_encoder_fp16_ixinfer_NHWC_msdeform.so ]]; then
  n_gemm=$(strings bevformer_encoder_fp16_ixinfer_NHWC_msdeform.so | grep -c ixinfer_gemm || true)
  if [[ "${n_gemm}" -gt 0 ]]; then
    ok "encoder SO embeds ixinfer_gemm (count=${n_gemm})"
  else
    note "encoder SO missing ixinfer_gemm — likely still float fused_matmul; rebuild"
  fi
fi

echo
if [[ "${fail}" -eq 0 ]]; then
  echo "RESULT: PASS (required checks ok; ${warn} warning(s))"
  exit 0
else
  echo "RESULT: FAIL — fix items above before expecting README_CN.md metrics"
  exit 1
fi
