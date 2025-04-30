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

import timm
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", 
                    type=str, 
                    required=True, 
                    help="Name of the model.")

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

    model = timm.create_model(args.model_name, checkpoint_path=args.weight)
    model.eval()

    dummy_input = torch.randn([32, 3, 288, 288])

    torch.onnx.export(
        model, 
        dummy_input, 
        args.output, 
        opset_version=13, 
        do_constant_folding=True, 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )

    print("Export onnx model successfully! ")

if __name__ == "__main__":
    main()