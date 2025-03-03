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

import argparse

import torch
from mmdeploy.utils import load_config
from mmdeploy.apis import build_task_processor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight", 
                    type=str, 
                    required=True, 
                    help="pytorch model weight.")

    parser.add_argument("--cfg", 
                    type=str, 
                    required=True, 
                    help="model config file.")
       
    parser.add_argument("--output", 
                    type=str, 
                    required=True, 
                    help="export onnx model path.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    deploy_cfg = 'deploy_default.py'
    model_cfg = args.cfg
    model_checkpoint = args.weight

    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    task_processor = build_task_processor(model_cfg, deploy_cfg, device='cpu')

    model = task_processor.build_pytorch_model(model_checkpoint)

    input_names = ['input']
    dynamic_axes = {'input': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 800, 800)

    torch.onnx.export(
        model, 
        dummy_input, 
        args.output, 
        input_names = input_names, 
        dynamic_axes = dynamic_axes, 
        opset_version=13
    )

    print("Export onnx model successfully! ")

if __name__ == '__main__':
    main()