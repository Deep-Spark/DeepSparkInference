#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

set -e

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
        EXIT_STATUS=1
    fi
}

# Run parameters
BSZ=${BSZ:-32}
IMGSZ=${IMGSZ:-224}
PRECISION=${PRECISION:-fp16}
DEVICE=${DEVICE:-0}
LIMIT=${LIMIT:-128}
COS_TARGET=${COS_TARGET:-0.999}
FORCE_BUILD=${FORCE_BUILD:-0}

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --imgsz) IMGSZ=${arguments[index]};;
      --device) DEVICE=${arguments[index]};;
      --limit) LIMIT=${arguments[index]};;
      -p|--precision) PRECISION=${arguments[index]};;
      --tgt) COS_TARGET=${arguments[index]};;
      -f|--force) FORCE_BUILD=1;;
    esac
done

PROJ_DIR=${PROJ_DIR:-.}
MODEL_DIR=${MODEL_DIR:-/data/models/dinov2-base}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-${MODEL_DIR}/checkpoints}
DATASETS_DIR=${DATASETS_DIR:-/data/dinov2-ixrt/testdata/imagenet-val-sub}
ORIGINE_MODEL=${ORIGINE_MODEL:-${MODEL_DIR}/model.onnx}
RUN_DIR=${RUN_DIR:-${PROJ_DIR}}

TAG="bs${BSZ}_${IMGSZ}"
STATIC_ONNX=${CHECKPOINTS_DIR}/dinov2_base_${TAG}.onnx
ENGINE_FILE=${CHECKPOINTS_DIR}/dinov2_base_${TAG}_${PRECISION}.engine

echo ====================== Model Info ======================
echo Model Name : dinov2-base
echo Onnx Path  : ${ORIGINE_MODEL}
echo Engine     : ${ENGINE_FILE}
echo Datasets   : ${DATASETS_DIR}
echo;

mkdir -p ${CHECKPOINTS_DIR}
cd ${PROJ_DIR}

step=1
echo [STEP ${step}] : Freeze ONNX + Build Engine
if [ ${FORCE_BUILD} -eq 1 ]; then
    rm -f ${STATIC_ONNX} ${ENGINE_FILE}
fi
if [ -f ${ENGINE_FILE} ] && [ -f ${STATIC_ONNX} ]; then
    echo "  "Build Skip, ${ENGINE_FILE} has been existed
else
    python3 ${RUN_DIR}/build_engine.py \
        --onnx ${ORIGINE_MODEL} \
        --out-dir ${CHECKPOINTS_DIR} \
        --batch ${BSZ} \
        --imgsz ${IMGSZ} \
        --precision ${PRECISION}
    echo "  "Generate ${ENGINE_FILE}
fi
let step++
echo;

echo [STEP ${step}] : Accuracy \(ixRT vs onnxruntime CPU FP32\)
ACC_ARGS=(
    --mode acc
    --engine ${ENGINE_FILE}
    --onnx ${STATIC_ONNX}
    --device ${DEVICE}
    --imgsz ${IMGSZ}
    --cos-target ${COS_TARGET}
    --limit ${LIMIT}
)
if [ -d "${DATASETS_DIR}" ]; then
    ACC_ARGS+=(--images ${DATASETS_DIR})
else
    echo "  "WARN: ${DATASETS_DIR} not found, fallback to random inputs
fi
python3 ${RUN_DIR}/inference.py "${ACC_ARGS[@]}"; check_status

exit ${EXIT_STATUS}
