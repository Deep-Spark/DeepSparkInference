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

import torch
import torchvision
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", 
                    type=str, 
                    required=True, 
                    help="Name of the model from torchvision.models.")

    parser.add_argument("--weight", 
                    type=str, 
                    required=True, 
                    help="pytorch model weight.")
    
    parser.add_argument("--output", 
                    type=str, 
                    required=True, 
                    help="export onnx model path.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f"Loading model: {args.model_name}...")
    if args.model_name == "resnest50":
        from resnest.torch import resnest50
        model = resnest50(pretrained=False)
    else:
        try:
            model = getattr(torchvision.models, args.model_name)(pretrained=False)
        except TypeError:
            # Fallback for models that do not accept 'pretrained' parameter
            model = getattr(torchvision.models, args.model_name)()

    state_dict = torch.load(args.weight)

    if args.model_name in ["densenet201", "densenet161", "densenet169", "densenet121"]:
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()

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
    
if __name__ == "__main__":
    main()
