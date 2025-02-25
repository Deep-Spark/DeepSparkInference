#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

set -x

pip3 install -r requirements.txt
cd ctc_decoder/swig && bash setup.sh
cd ../../

# tar -zxvf 20211025_conformer_exp.tar.gz

# Get Onnx Model
cd wenet
python3 wenet/bin/export_onnx_gpu.py                          \
    --config ../20211025_conformer_exp/train.yaml             \
    --checkpoint ../20211025_conformer_exp/final.pt           \
    --batch_size 24                                           \
    --seq_len 384                                             \
    --beam 4                                                  \
    --cmvn_file ../20211025_conformer_exp/global_cmvn         \
    --output_onnx_dir ../
cd ..

# Use onnxsim optimize onnx model
onnxsim encoder_bs24_seq384_static.onnx encoder_bs24_seq384_static_opt.onnx
python3 alter_onnx.py --batch_size 24 --path encoder_bs24_seq384_static_opt.onnx

# Need to unzip aishell to the current directory. For details, refer to data.list
tar -zxvf aishell.tar.gz
