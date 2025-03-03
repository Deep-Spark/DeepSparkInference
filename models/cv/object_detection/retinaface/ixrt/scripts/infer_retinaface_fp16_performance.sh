#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

# Run paraments
BSZ=32
TGT=-1
WARM_UP=3
LOOP_COUNT=20
RUN_MODE=FPS
PRECISION=float16

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --tgt) TGT=${arguments[index]};;
    esac
done

PROJ_DIR=${PROJ_DIR:-"."}
DATASETS_DIR="${DATASETS_DIR}"
CHECKPOINTS_DIR="${PROJ_DIR}"
RUN_DIR="${PROJ_DIR}"
CONFIG_DIR="${PROJ_DIR}/config/RETINAFACE_CONFIG"
source ${CONFIG_DIR}
ORIGINE_MODEL=${CHECKPOINTS_DIR}/${ORIGINE_MODEL}

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo CONFIG_DIR : ${CONFIG_DIR}
echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Onnx Path : ${ORIGINE_MODEL}
SIM_MODEL=${ORIGINE_MODEL}

step=0

# Quant Model
if [ $PRECISION == "int8" ];then
    let step++
    echo;
    echo [STEP ${step}] : Quant Model
    if [[ -z ${QUANT_EXIST_ONNX} ]];then
        QUANT_EXIST_ONNX=$CHECKPOINTS_DIR/quantized_${MODEL_NAME}.onnx
    fi
    if [[ -f ${QUANT_EXIST_ONNX} ]];then
        SIM_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Quant Model Skip, ${QUANT_EXIST_ONNX} has been existed
    else
        python3 ${RUN_DIR}/quant.py            \
            --model ${SIM_MODEL}               \
            --model_name ${MODEL_NAME}         \
            --dataset_dir ${DATASETS_DIR}      \
            --observer ${QUANT_OBSERVER}       \
            --disable_quant_names ${DISABLE_QUANT_LIST[@]} \
            --save_dir $CHECKPOINTS_DIR        \
            --bsz   ${QUANT_BATCHSIZE}         \
            --step  ${QUANT_STEP}              \
            --seed  ${QUANT_SEED}              \
            --imgsz ${IMGSIZE}
        SIM_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Generate ${SIM_MODEL}
    fi
fi

# Build Engine
let step++
echo;
echo [STEP ${step}] : Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_engine.py          \
        --precision ${PRECISION}                \
        --model ${SIM_MODEL}                    \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}/inference.py     \
    --engine_file=${ENGINE_FILE}    \
    --datasets_dir=${DATASETS_DIR}  \
    --imgsz=${IMGSIZE}              \
    --save_folder="${CHECKPOINTS_DIR}/widerface_txt_${BSZ}" \
    --pred_path="${CHECKPOINTS_DIR}/widerface_txt_${BSZ}"  \
    --gt=${GT_DIR}                  \
    --warm_up=${WARM_UP}            \
    --loop_count ${LOOP_COUNT}      \
    --test_mode ${RUN_MODE}         \
    --fps_target ${TGT}             \
    --bsz ${BSZ}; check_status

exit ${EXIT_STATUS}