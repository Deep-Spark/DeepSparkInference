# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
#!/bin/bash
EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

PROJ_DIR=$(cd $(dirname $0);cd ../; pwd)
DATASETS_DIR="${PROJ_DIR}/data/coco"
COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
EVAL_DIR=${DATASETS_DIR}/images/val2017
CHECKPOINTS_DIR="${PROJ_DIR}/data"
RUN_DIR="${PROJ_DIR}"
ORIGINE_MODEL=${CHECKPOINTS_DIR}
DISABLE_NAMES=('/model.22/Concat' '/model.22/Concat_1' '/model.22/Concat_2' '/model.22/Reshape' '/model.22/Reshape_1' '/model.22/Reshape_2' '/model.22/Concat_3' '/model.22/Split' '/model.22/dfl/Reshape' '/model.22/dfl/Transpose' '/model.22/dfl/Softmax' '/model.22/dfl/Transpose_1' '/model.22/dfl/conv/Conv' '/model.22/dfl/Reshape_1' '/model.22/Slice' '/model.22/Slice_1' '/model.22/Sub' '/model.22/Add_1' '/model.22/Add_2' '/model.22/Div_1' '/model.22/Sub_1' '/model.22/Concat_4' '/model.22/Mul_2' '/model.22/Sigmoid' '/model.22/Concat_5')

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo ====================== Model Info ======================
echo Model Name : yolov8n
echo Onnx Path : ${ORIGINE_MODEL}

BATCH_SIZE=32
CURRENT_MODEL=${CHECKPOINTS_DIR}/yolov8n.onnx

# quant
FINAL_MODEL=${CHECKPOINTS_DIR}/quantized_yolov8n_bs${BATCH_SIZE}.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Quantize Skip, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}/quant.py             \
        --model_name "YOLOV8N"       \
        --model ${CURRENT_MODEL}            \
        --bsz ${BATCH_SIZE}                 \
        --dataset_dir ${EVAL_DIR}           \
        --ann_file ${COCO_GT}               \
        --observer "hist_percentile"        \
        --save_quant_model ${FINAL_MODEL}   \
        --disable_quant_names '/model.22/Concat' '/model.22/Concat_1' '/model.22/Concat_2' '/model.22/Reshape' '/model.22/Reshape_1' '/model.22/Reshape_2' '/model.22/Concat_3' '/model.22/Split' '/model.22/dfl/Reshape' '/model.22/dfl/Transpose' '/model.22/dfl/Softmax' '/model.22/dfl/Transpose_1' '/model.22/dfl/conv/Conv' '/model.22/dfl/Reshape_1' '/model.22/Slice' '/model.22/Slice_1' '/model.22/Sub' '/model.22/Add_1' '/model.22/Add_2' '/model.22/Div_1' '/model.22/Sub_1' '/model.22/Concat_4' '/model.22/Mul_2' '/model.22/Sigmoid' '/model.22/Concat_5' \
        --imgsz 640                          
    echo "  "Generate ${FINAL_MODEL}
fi
CURRENT_MODEL=${FINAL_MODEL}

# Build Engine
echo Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/yolov8n_int8.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_engine.py          \
        --precision int8                        \
        --model ${CURRENT_MODEL}                \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
echo Inference
RUN_BATCH_SIZE=32
python3 ${RUN_DIR}/inference.py                 \
    --model_engine ${ENGINE_FILE}                \
    --warm_up 2                                 \
    --bsz ${RUN_BATCH_SIZE}                         \
    --imgsz 640                              \
    --datasets ${DATASETS_DIR}               \
    --acc_target 0.3                     
exit ${EXIT_STATUS}
