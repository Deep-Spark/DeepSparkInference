#!/bin/bash

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

batchsize=64
model_path="dinov2_vits14_pretrain.onnx"
datasets_path=${DATASETS_DIR}

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) batchsize=${arguments[index]};;
    esac
done

echo "batch size is ${batchsize}"

# build engine
cd ./dinov2
python3 ${RUN_DIR}build_engine.py                     \
    --model_path ${model_path}              \
    --input input:${batchsize},3,224,224    \
    --precision fp16                        \
    --engine_path dinov2_vits14_pretrain_bs_${batchsize}_fp16.so


# inference
python3 dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights dinov2_vits14_pretrain_bs_${batchsize}_fp16.so \
    --output-dir ./eval_output/linear_vits14 \
    --ngpus 1 \
    --batch-size ${batchsize} \
    --train-dataset ImageNet:split=TRAIN:root=${IMAGENET_1K}:extra=${IMAGENET_1K}/extra \
    --val-dataset ImageNet:split=VAL:root=${IMAGENET_1K}:extra=${IMAGENET_1K}/extra