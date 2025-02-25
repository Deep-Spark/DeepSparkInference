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

batchsize=24
seqlen=384
model_path="encoder_bs24_seq384_static_opt_matmul.onnx"

# build engine
python3 build_engine.py                                                                     \
    --model_path ${model_path}                                                              \
    --input speech:${batchsize},${seqlen},80 speech_lengths:${batchsize}                    \
    --precision fp16                                                                        \
    --engine_path encoder_bs${batchsize}_seq${seqlen}_fp16.so

# inference
python3 inference.py                                          \
  --engine encoder_bs${batchsize}_seq${seqlen}_fp16.so        \
  --input speech speech_lengths                               \
  --label text                                                \
  --config train.yaml                                         \
  --test_data data.list                                       \
  --dict lang_char.txt                                        \
  --mode ctc_greedy_search                                    \
  --batch_size ${batchsize}                                   \
  --seq_len ${seqlen}                                         \
  --result_file conformer_output_log