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

cd ../CenterPoint/

# # build engine with igie-exec, make sure you have exported the rpn_opt.onnx, see README.md
# igie-exec --model_path onnx_model/rpn_opt.onnx \
#         --input input.1:4,64,512,512 \
#         --precision fp16 \
#         --engine_path rpn_opt.so \
#         --just_export True

# inference
python3 tools/igie_test.py \
        configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini_igie.py \
        --work_dir dataset/nuscenes  \
        --checkpoint ./latest.pth