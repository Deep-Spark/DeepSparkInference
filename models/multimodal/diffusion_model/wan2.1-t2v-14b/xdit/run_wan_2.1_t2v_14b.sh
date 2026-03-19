#!/bin/bash
set -x
export WORD_RANK_SUPPORT_TP=1
export ATTN_OPT_LEVEL=2 #xdit >=0.4.5
export ENABLE_IXFORMER_SAGEATTN=1 #xdit ==0.4.4
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="wan2.1_t2v_example.py"
MODEL_ID="/data/nlp/Wan2.1-T2V-14B-Diffusers/"
INFERENCE_STEP=20

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 480 --width 832 --num_frames 33 --seed 33 "

# CogVideoX parallel configuration
N_GPUS=4
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 1 --tensor_parallel_degree  2" 
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# ENABLE_TILING="--enable_tiling"
# MODEL_OFFLOAD="--enable_model_cpu_offload"
ENABLE_CACHE="--use_teacache"
COMPILE_FLAG="--use_torch_compile"
#ENABLE_W8A8="--use_w8a8_linear"

torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$ENABLE_W8A8 \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A rainy night in a dense cyberpunk market, neon kanji signs flicker overhead. The camera starts shoulder-height behind a hooded courier, steadily tracking forward as he weaves through crowds of holographic umbrellas. Volumetric pink-blue backlight cuts through steam vents, puddles mirror the glow. Lens flare, shallow depth of field. Moody, Blade-Runner vibe." \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
$ENABLE_TILING \
$ENABLE_CACHE \
$COMPILE_FLAG \
$CFG_ARGS
