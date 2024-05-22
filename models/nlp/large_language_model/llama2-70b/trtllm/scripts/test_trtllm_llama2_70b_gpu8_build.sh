#!/bin/bash

EXIT_STATUS=0
LOG_LEVEL=info
BS=${BS:-1}
DTYPE=${DTYPE:-"float16"}

PROJECT_DIR="./"

MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/data/llama2-70b-chat"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_DIR}/checkpoints/"}

export TLLM_LOG_LEVEL=${LOG_LEVEL}
export PLUGIN_DTYPE="float16"

check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

python3 ${PROJECT_DIR}/build.py \
--log_level ${LOG_LEVEL} \
--dtype ${DTYPE} \
--model_dir ${MODEL_DIR} \
--remove_input_padding \
--use_gpt_attention_plugin float16 --use_gemm_plugin float16 \
--enable_context_fmha \
--world_size 8 \
--tp_size 8 \
--output_dir ${OUTPUT_DIR} "$@"; check_status
exit ${EXIT_STATUS}


