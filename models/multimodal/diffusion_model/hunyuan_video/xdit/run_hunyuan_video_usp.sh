#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH
export NCCL_USE_HIGHPRIORITYWARP=1
export ENABLE_IXFORMER_INFERENCE=1
export ATTN_OPT_LEVEL=2 #xdit >=0.4.5
export ENABLE_IXFORMER_SAGEATTN=1  #使用 sageattention,#xdit ==0.4.4

SCRIPT="hunyuan_video_usp_example.py"
MODEL_ID="/data/nlp/HunyuanVideo/"
INFERENCE_STEP=50
mkdir -p ./results

TASK_ARGS="--height 720 --width 1280 --num_frames 133 --guidance_scale 5.0"

N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 4 --ring_degree 2"
ENABLE_TILING="--enable_tiling"
ENABLE_MODEL_CPU_OFFLOAD="--enable_model_cpu_offload"
COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A cat walks on the grass, realistic" \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$ENABLE_MODEL_CPU_OFFLOAD \
$COMPILE_FLAG
