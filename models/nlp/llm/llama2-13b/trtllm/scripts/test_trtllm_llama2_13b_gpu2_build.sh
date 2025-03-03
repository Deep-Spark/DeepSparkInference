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
BUILD_TIME_TARGET=${BUILD_TIME_TARGET:-72}

PROJECT_DIR="./"

MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/data/llama2-13b-chat"}
ENGINE_DIR=${ENGINE_DIR:-"${PROJECT_DIR}"}
CHECKPOINT_DIR="${ENGINE_DIR}/checkpoints"

export TLLM_LOG_LEVEL=${LOG_LEVEL}
export PLUGIN_DTYPE="float16"

check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}


python3 convert_checkpoint.py \
--model_dir ${MODEL_DIR} \
--output_dir ${CHECKPOINT_DIR} \
--tp_size 2 \
--workers 2 \
--dtype ${DTYPE}


# best(build engine time: 50) is 70% of target
trtllm-build \
--log_level ${LOG_LEVEL}    \
--max_batch_size ${BS}      \
--checkpoint_dir ${CHECKPOINT_DIR} \
--remove_input_padding enable \
--context_fmha enable \
--workers 2 \
--total_build_time_target ${BUILD_TIME_TARGET} \
--output_dir ${ENGINE_DIR} "$@"; check_status
exit ${EXIT_STATUS}