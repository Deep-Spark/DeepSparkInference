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

set -euo pipefail


EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    echo "fails"
    EXIT_STATUS=1
    fi
}

current_path=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

PROJECT_DIR=${current_path}/..
DATA_DIR=${current_path}/../aishell_test_data/test
MODEL_DIR=${current_path}/../conformer_checkpoints

export Accuracy=${Accuracy:=350}

cd ${PROJECT_DIR}


echo "Step1.Export Onnx From Checkpoints!"
python3 convert2onnx.py \
    --model_name "Conformer" \
    --model_path=${MODEL_DIR}/final.pt                          \
    --onnx_path=${MODEL_DIR}/conformer_encoder_fusion.onnx      \
    --batch_size=24

echo "Step2.Build Engine!"
python3 build_engine.py \
    --model_name "Conformer" \
    --onnx_path=${MODEL_DIR}/conformer_encoder_fusion.onnx        \
    --engine_path=${MODEL_DIR}/conformer_encoder_fusion.engine    \
    --max_batch_size=24  \
    --max_seq_len=1500

echo "Step3.Inference(Test QPS)!"
python3 ixrt_inference_performance.py \
    --infer_type fp16 \
    --batch_size ${BATCH_SIZE:=24} \
    --data_dir ${DATA_DIR}  \
    --model_dir ${MODEL_DIR}
