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
LOG_LEVEL=info
BS=${BS:-1}
DTYPE=${DTYPE:-"float16"}

PROJECT_DIR=$(dirname "$(dirname "$(dirname "$(cd "$(dirname "$0")"; pwd)")")")

MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/data/llama2/llama2-13b-chat"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_DIR}/tmp/trtllm/llama2/13B/trt_engines/fp16/2-gpu/"}

echo PROJECT_DIR : ${PROJECT_DIR}

export TLLM_LOG_LEVEL=${LOG_LEVEL}
export PLUGIN_DTYPE="float16"

check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

# best(build engine time: 223.33) is 95% of target
python3 ${PROJECT_DIR}/llama2/llama2_13b_gpu2/build.py \
--log_level ${LOG_LEVEL} \
--dtype ${DTYPE} \
--model_dir ${MODEL_DIR} \
--remove_input_padding \
--use_gpt_attention_plugin float16 --use_gemm_plugin float16 \
--enable_context_fmha \
--disable_xqa \
--world_size 2 \
--tp_size 2 \
--total_build_time_target 235.1 \
--output_dir ${OUTPUT_DIR} "$@"; check_status
exit ${EXIT_STATUS}
