#!/bin/bash

# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
export PYTHONPATH=/workspace/deepsparkinference/models/cv/classification/dinov2/igie/dinov2:${PYTHONPATH}
cd dinov2
python3 dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights dinov2_vits14_pretrain_bs_64_fp16.so \
    --output-dir ./eval_output/linear_vits14_tvm \
    --ngpus 1 \
    --batch-size 64 \
    --train-dataset ImageNet:split=TRAIN:root=${IMAGENET_1K}:extra=${IMAGENET_1K}/extra \
    --val-dataset ImageNet:split=VAL:root=${IMAGENET_1K}:extra=${IMAGENET_1K}/extra