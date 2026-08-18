#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
apt update
apt-get install sox libsox-dev
pip3 install -r requirements.txt
pip3 install onnxruntime==1.18.0 transformers==4.49.0
cp -r /mnt/deepspark/data/repos/CosyVoice ./
cd CosyVoice
git checkout 074ca6dc9e80a2f42
rm -rf pretrained_models
mkdir -p pretrained_models
ln -s /mnt/deepspark/data/checkpoints/CosyVoice2-0.5B pretrained_models/

# python3 example.py

cd ..
mkdir -p /root/.cache/modelscope/hub/iic
ln -s /mnt/deepspark/data/checkpoints/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch /root/.cache/modelscope/hub/iic/
cp -r /mnt/deepspark/data/repos/CV3-Eval ./
cd CV3-Eval
git checkout 68b7682a410d048
mv ../CosyVoice ./
cp ../file_utils.py CosyVoice/cosyvoice/utils/
cp ../processor.py CosyVoice/cosyvoice/dataset/
cp ../get_infer_wavs.py scripts/
cp ../inference.sh scripts/
cp ../run_score_wer.sh scripts/
cp ../cal_wer.sh utils/
cp -f ../run_wer.py utils/
cp -f ../eval_speaker_similarity.py scripts/
cp ../run_inference_fp16_eval.sh ./
bash run_inference_fp16_eval.sh