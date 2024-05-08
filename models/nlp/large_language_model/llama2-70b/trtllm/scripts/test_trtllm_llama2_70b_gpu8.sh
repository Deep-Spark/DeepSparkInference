#!/bin/bash

EXIT_STATUS=0
LOG_LEVEL=info
BS=${BS:-1}
DTYPE=${DTYPE:-"float16"}

PROJECT_DIR="./"

DATASET_DIR=${DATASET_DIR:-"${PROJECT_DIR}/data/datasets_cnn_dailymail"}
MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/data/llama2-70b-chat"}
ENGINE_DIR=${ENGINE_DIR:-"${PROJECT_DIR}/checkpoints/"}

export TLLM_LOG_LEVEL=${LOG_LEVEL}
export PLUGIN_DTYPE="float16"

check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}


export TASK_DATA_PATH=${DATASET_DIR}

# target is 95% of best (load engine time: 14.65, rouge1: 29.19, tps: 18.59)
mpirun -n 8 --allow-run-as-root \
python3 ${PROJECT_DIR}/summarize.py \
--test_trt_llm \
--log_level ${LOG_LEVEL} \
--batch_size ${BS}  \
--data_type ${DTYPE} \
--hf_model_dir ${MODEL_DIR} \
--tokenizer_dir ${MODEL_DIR} \
--tokenizer_type "llama" \
--engine_dir ${ENGINE_DIR}  \
--target_load_engine_time 15.4 \
--tensorrt_llm_rouge1_threshold 27.73    \
--target_tps 17.66  \
--use_py_session "$@"; check_status
exit ${EXIT_STATUS}
