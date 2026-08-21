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
WARM_UP=${WARM_UP:-}
ITERS=${ITERS:-}
FPS_TARGET=${FPS_TARGET:--1}
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
      --warmup) WARM_UP=${arguments[index]};;
      --iters) ITERS=${arguments[index]};;
      -p|--precision) PRECISION=${arguments[index]};;
      --tgt) FPS_TARGET=${arguments[index]};;
      -f|--force) FORCE_BUILD=1;;
    esac
done

# bs=1 measures latency, so it needs more iterations to stabilize
if [[ ${BSZ} == 1 ]]; then
    WARM_UP=${WARM_UP:-50}
    ITERS=${ITERS:-500}
else
    WARM_UP=${WARM_UP:-20}
    ITERS=${ITERS:-200}
fi
echo "WARM_UP=${WARM_UP} ITERS=${ITERS} for bsz=${BSZ}"

PROJ_DIR=${PROJ_DIR:-.}
MODEL_DIR=${MODEL_DIR:-/data/models/dinov2-base}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-${MODEL_DIR}/checkpoints}
ORIGINE_MODEL=${ORIGINE_MODEL:-${MODEL_DIR}/model.onnx}
RUN_DIR=${RUN_DIR:-${PROJ_DIR}}

TAG="bs${BSZ}_${IMGSZ}"
STATIC_ONNX=${CHECKPOINTS_DIR}/dinov2_base_${TAG}.onnx
ENGINE_FILE=${CHECKPOINTS_DIR}/dinov2_base_${TAG}_${PRECISION}.engine

echo ====================== Model Info ======================
echo Model Name : dinov2-base
echo Onnx Path  : ${ORIGINE_MODEL}
echo Engine     : ${ENGINE_FILE}
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

echo [STEP ${step}] : Performance
python3 ${RUN_DIR}/inference.py \
    --mode perf \
    --engine ${ENGINE_FILE} \
    --device ${DEVICE} \
    --imgsz ${IMGSZ} \
    --warmup ${WARM_UP} \
    --iters ${ITERS} \
    --fps-target ${FPS_TARGET}; check_status

exit ${EXIT_STATUS}
