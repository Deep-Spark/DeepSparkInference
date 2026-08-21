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

set -ex

# Optional system deps (Pillow / OpenCV may need libGL on bare hosts)
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"' || true)
if [[ ${ID} == "ubuntu" ]]; then
    apt-get update && apt-get install -y libgl1-mesa-glx || true
elif [[ ${ID} == "centos" ]]; then
    yum install -y mesa-libGL || true
fi

pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# onnxsim has no py3.12 wheel on some images; build_engine.py falls back to
# onnxruntime BASIC constant folding, so a failure here is not fatal.
pip3 install onnxsim -i https://pypi.tuna.tsinghua.edu.cn/simple || true

MODEL_DIR=${MODEL_DIR:-/data/models/dinov2-base}
mkdir -p "${MODEL_DIR}" "${MODEL_DIR}/checkpoints"

ONNX_PATH="${MODEL_DIR}/model.onnx"
if [ ! -f "${ONNX_PATH}" ]; then
    # Prefer local mirror / shared cache; fall back to HuggingFace mirror
    if [ -f /mnt/deepspark/data/checkpoints/dinov2-base/model.onnx ]; then
        cp /mnt/deepspark/data/checkpoints/dinov2-base/model.onnx "${ONNX_PATH}"
        cp /mnt/deepspark/data/checkpoints/dinov2-base/config.json "${MODEL_DIR}/" 2>/dev/null || true
        cp /mnt/deepspark/data/checkpoints/dinov2-base/preprocessor_config.json "${MODEL_DIR}/" 2>/dev/null || true
    else
        wget -O "${ONNX_PATH}" \
            https://hf-mirror.com/onnx-community/dinov2-base/resolve/main/onnx/model.onnx
        wget -O "${MODEL_DIR}/config.json" \
            https://hf-mirror.com/onnx-community/dinov2-base/resolve/main/config.json
        wget -O "${MODEL_DIR}/preprocessor_config.json" \
            https://hf-mirror.com/onnx-community/dinov2-base/resolve/main/preprocessor_config.json
    fi
fi

# Expected size: 346627111 bytes
ls -l "${ONNX_PATH}"
