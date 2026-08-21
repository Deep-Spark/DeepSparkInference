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

# Downstream leave-one-out 1-NN on ImageFolder features (DINOv2 has no classifier head)
BSZ=${BSZ:-32}
IMGSZ=${IMGSZ:-224}
PRECISION=${PRECISION:-fp16}
DEVICE=${DEVICE:-0}
PER_CLASS=${PER_CLASS:-0}
LIMIT=${LIMIT:-0}
FORCE_BUILD=${FORCE_BUILD:-0}

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
      --per-class) PER_CLASS=${arguments[index]};;
      --limit) LIMIT=${arguments[index]};;
      -p|--precision) PRECISION=${arguments[index]};;
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

if [ ! -d "${DATASETS_DIR}" ]; then
    echo "ERROR: DATASETS_DIR=${DATASETS_DIR} not found (need ImageFolder layout)"
    exit 1
fi

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
fi
let step++
echo;

echo [STEP ${step}] : Downstream kNN
KNN_ARGS=(
    --mode knn
    --engine ${ENGINE_FILE}
    --onnx ${STATIC_ONNX}
    --images ${DATASETS_DIR}
    --device ${DEVICE}
    --imgsz ${IMGSZ}
)
if [ ${PER_CLASS} -gt 0 ]; then
    KNN_ARGS+=(--per-class ${PER_CLASS})
fi
if [ ${LIMIT} -gt 0 ]; then
    KNN_ARGS+=(--limit ${LIMIT})
fi
python3 ${RUN_DIR}/inference.py "${KNN_ARGS[@]}"; check_status

exit ${EXIT_STATUS}
