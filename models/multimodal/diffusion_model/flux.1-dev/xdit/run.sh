# set -x
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PYTHONPATH=$PWD:$PYTHONPATH


export ENABLE_IXFORMER_INFERENCE=1
export ATTN_OPT_LEVEL=2 #xdit >=0.4.5
export ENABLE_IXFORMER_SAGEATTN=1  #使用 sageattention,#xdit ==0.4.4
export ENABLE_IXFORMER_W8A8LINEAR=1

# Select the model type
export MODEL_TYPE="Flux"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(    
    ["Flux"]="flux_example.py /home/data/flux___1-schnell/ 28"   
)

echo ${MODEL_CONFIGS[$MODEL_TYPE]}

# if [ -v MODEL_CONFIGS[$MODEL_TYPE] ] ; then
if [ -n "${MODEL_CONFIGS[$MODEL_TYPE]+_}" ]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi



# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 3.5"

# cache args
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
N_GPUS=2
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 1"

# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"


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

