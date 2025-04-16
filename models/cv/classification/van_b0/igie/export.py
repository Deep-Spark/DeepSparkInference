# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
from mmcls.apis import init_model

class Model(torch.nn.Module):
    def __init__(self, config_file, checkpoint_file):
        super().__init__()
        self.model = init_model(config_file, checkpoint_file, device="cpu") 
  
    def forward(self, x):
        feat = self.model.backbone(x)
        feat = self.model.neck(feat[0])
        out_head = self.model.head.fc(feat)
        return out_head
    
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

    config_file = args.cfg
    checkpoint_file = args.weight
    model = Model(config_file, checkpoint_file).eval()

    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model, 
        dummy_input, 
        args.output, 
        input_names = input_names, 
        dynamic_axes = dynamic_axes, 
        output_names = output_names,
        opset_version=13
    )

    print("Export onnx model successfully! ")

if __name__ == '__main__':
    main()

