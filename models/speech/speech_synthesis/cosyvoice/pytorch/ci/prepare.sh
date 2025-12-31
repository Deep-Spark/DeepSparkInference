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
git clone https://github.com/FunAudioLLM/CV3-Eval.git
cd CV3-Eval
pip3 install -r requirements.txt
cp ../get_infer_wavs.py scripts/
cp ../inference.sh scripts/

# if you want to run eval for en/hrad_en set, please add the following command
# cp -f ../run_wer.py utils/

cp ../run_inference_fp16_eval.sh ./
bash run_inference_fp16_eval.sh