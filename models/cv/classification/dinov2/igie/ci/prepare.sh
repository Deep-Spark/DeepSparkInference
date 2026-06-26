#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cp -r /mnt/deepspark/data/3rd_party/dinov2 ./
cp -r eval dinov2/dinov2
cp dinov2-patch/* dinov2/
pip3 install omegaconf==2.3.0 fvcore iopath submitit==1.5.4 torchmetrics

cd dinov2
ln -s /mnt/deepspark/data/checkpoints/dinov2_vits14_pretrain.pth dinov2_vits14_pretrain.pth

python3 prepare_datasets.py

python3 export.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights dinov2_vits14_pretrain.pth \
    --onnx-path dinov2_vits14_pretrain.onnx \
    --input-size 224 \
    --n-last-blocks 4 \
    --device cuda

export batchsize=64
python3 build_engine.py                     \
    --model_path dinov2_vits14_pretrain.onnx              \
    --input input:${batchsize},3,224,224    \
    --precision fp16                        \
    --engine_path dinov2_vits14_pretrain_bs_${batchsize}_fp16.so