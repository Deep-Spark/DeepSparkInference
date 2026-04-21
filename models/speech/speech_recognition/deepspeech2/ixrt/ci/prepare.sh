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

set -x

pip3 install librosa psutil pysoundfile pytest requests tensorboardX editdistance textgrid onnxsim paddlespeech_ctcdecoders paddleaudio paddlespeech
pip3 install numpy==1.23.5

mkdir -p checkpoints
cp /root/data/checkpoints/deepspeech2.onnx checkpoints/
cp /root/data/checkpoints/common_crawl_00.prune01111.trie.klm checkpoints/


OPTIMIER_FILE=/root/data/3rd_party/iluvatar-corex-ixrt/tools/optimizer/optimizer.py
echo "Build engine!"
python3 modify_model_to_dynamic.py --static_onnx checkpoints/deepspeech2.onnx --dynamic_onnx checkpoints/deepspeech2_dynamic.onnx
python3 ${OPTIMIER_FILE}  --onnx checkpoints/deepspeech2_dynamic.onnx --model_type rnn --not_sim
python3 build_engine.py \
    --model_name deepspeech2 \
    --onnx_path checkpoints/deepspeech2_dynamic_end.onnx \
    --engine_path checkpoints/deepspeech2.engine
