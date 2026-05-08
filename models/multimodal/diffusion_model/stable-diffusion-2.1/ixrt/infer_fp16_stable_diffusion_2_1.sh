#!/bin/bash

# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

# Run paraments
BSZ=1
TGT=-1

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --tgt) TGT=${arguments[index]};;
    esac
done

IXRT_DIR=$(python3 -c "import ixrt; print(ixrt.__file__)" | xargs dirname)

PROJ_DIR=./
CHECKPOINTS_DIR="${PROJ_DIR}checkpoints/stable_diffusion_2_1_ixrt"
DATASETS_DIR="${PROJ_DIR}datasets/stable_diffusion_2_1_ixrt"
echo "CHECKPOINTS_DIR: ${CHECKPOINTS_DIR}"
echo "DATASETS_DIR: ${DATASETS_DIR}"

# Check ONNX files
for f in unet.onnx clip.onnx vae.onnx; do
    if [ ! -f "${CHECKPOINTS_DIR}/${f}" ]; then
        echo "Error: ${f} not found in ${CHECKPOINTS_DIR}"
        exit 1
    fi
done

# Build engines from original ONNX
echo "[Step 1] Building clip engine..."
ixrtexec --onnx=${CHECKPOINTS_DIR}/clip.onnx \
--opt_shape=input_ids:1x77 \
--min_shape=input_ids:1x77 \
--max_shape=input_ids:4x77 \
--save_engine ${CHECKPOINTS_DIR}/clip.engine \
--precision fp16 \
--plugin "$IXRT_DIR/lib/libixrt_plugin.so"
check_status

echo "[Step 2] Building unet engine..."
ixrtexec --onnx=${CHECKPOINTS_DIR}/unet.onnx \
--opt_shape=sample:2x4x64x64,encoder_hidden_states:2x77x1024,timestep:1 \
--min_shape=sample:1x4x64x64,encoder_hidden_states:1x77x1024,timestep:1 \
--max_shape=sample:8x4x64x64,encoder_hidden_states:8x77x1024,timestep:1 \
--save_engine ${CHECKPOINTS_DIR}/unet.engine \
--precision fp16 \
--plugin "$IXRT_DIR/lib/libixrt_plugin.so"
check_status

echo "[Step 3] Building vae engine..."
ixrtexec --onnx=${CHECKPOINTS_DIR}/vae.onnx \
--opt_shape=latent:1x4x64x64 \
--min_shape=latent:1x4x64x64 \
--max_shape=latent:4x4x64x64 \
--save_engine ${CHECKPOINTS_DIR}/vae.engine \
--precision fp16 \
--plugin "$IXRT_DIR/lib/libixrt_plugin.so"
check_status

if [ $EXIT_STATUS -eq 0 ]; then
    echo "All build engine completed successfully!"
    echo "All engines built successfully!"
else
    echo "Some build engine failed!"
    echo "Engine build failed!"
    exit 1
fi

# infer model
python3 inference.py  \
--prompt-file ${DATASETS_DIR}/stable_diffusion_2_1_input.txt   \
-v --version 2.1 --seed 456123 --batch-size ${BSZ} --width 512 --height 512 --denoising-steps 20 --num-warmup-runs 1 \
--engine-path ${CHECKPOINTS_DIR}     \
--perform-target ${TGT}              \
--correct-image-path  ${DATASETS_DIR}/stable_diffusion_2_1_result.png  \
--ddim-scheduler-path  ${DATASETS_DIR}/scheduler_config.json  \
--clip-tokenizer-path  ${DATASETS_DIR}/tokenizer
check_status

if [ $EXIT_STATUS -eq 0 ]; then
    echo "infer successfully!"
else
    echo "infer failed!"
    exit 1
fi