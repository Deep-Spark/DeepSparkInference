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

echo "start run $0"

EXIT_STATUS=0
LOG_LEVEL=${LOG_LEVEL:-INFO}
BS=${BS:-1}
DTYPE=${DTYPE:-"float16"}
LOAD_TIME_TARGET=${LOAD_TIME_TARGET:-16}
TPS_TARGET=${TPS_TARGET:-32.2}

PROJECT_DIR="./"

DATASET_DIR=${DATASET_DIR:-"${PROJECT_DIR}/data/datasets_cnn_dailymail"}
MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/data/llama2-13b-chat"}
ENGINE_DIR=${ENGINE_DIR:-"${PROJECT_DIR}"}

export TLLM_LOG_LEVEL=${LOG_LEVEL}
export PLUGIN_DTYPE="float16"

check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

export TASK_DATA_PATH=${DATASET_DIR}

# target is 80% of best (load engine time: 11, rouge1: 29.54, tps: 33.9)
mpirun -n 2 --allow-run-as-root \
python3 ${PROJECT_DIR}/summarize.py \
--test_trt_llm \
--log_level ${LOG_LEVEL} \
--batch_size ${BS}  \
--data_type ${DTYPE} \
--hf_model_dir ${MODEL_DIR} \
--tokenizer_dir ${MODEL_DIR} \
--tokenizer_type "llama" \
--engine_dir ${ENGINE_DIR}  \
--tensorrt_llm_rouge1_threshold 27.7    \
--target_load_engine_time ${LOAD_TIME_TARGET} \
--target_tps ${TPS_TARGET}  \
--use_py_session "$@"; check_status
exit ${EXIT_STATUS}