# set -x
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PYTHONPATH=$PWD:$PYTHONPATH

#多ring 没提升
# export NCCL_USE_HIGHPRIORITYWARP=1

export ENABLE_IXFORMER_INFERENCE=1
# export ATTN_OPT_LEVEL=2
export ENABLE_IXFORMER_W8A8LINEAR=0

# Select the model type
SCRIPT=hunyuandit_example.py
MODEL_ID=/data/nlp/HunyuanDiT-v1.2-Diffusers/
INFERENCE_STEP=20

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 3.5"

# cache args
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

N_GPUS=2
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1 --tensor_parallel_degree 1 --data_parallel_degree 1"
CFG_ARGS="--use_cfg_parallel"

torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "brown dog laying on the ground with a metal bowl in front of him." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS \

