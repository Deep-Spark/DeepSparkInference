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
cp -r /mnt/deepspark/data/repos/detr/* ./

# change images size
sed -i '105 s/size = get_size(image.size, size, max_size)/size = (800, 800)/' ./datasets/transforms.py

pip3 install --no-build-isolation -r requirements.txt
pip3 install onnxsim
pip install -U pycocotools
mkdir -p /root/.cache/torch/hub/checkpoints/
ln -s /mnt/deepspark/data/checkpoints/resnet50-0676ba61.pth  /root/.cache/torch/hub/checkpoints/
python3 export.py --no_aux_loss --eval --resume detr-r50-e632da11.pth --coco_path ./coco

onnxsim detr.onnx detr_opt.onnx