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
from pathlib import Path
from transformers.onnx import export
from transformers.models.clip import CLIPOnnxConfig
from transformers import CLIPProcessor, CLIPModel

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output", 
                    type=str, 
                    required=True, 
                    help="export onnx model path.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    checkpoint = "clip-vit-base-patch32"
    feature_extractor = CLIPProcessor.from_pretrained(checkpoint)
    model = CLIPModel.from_pretrained(checkpoint)
    
    save_path = Path(args.output)
    onnx_config = CLIPOnnxConfig(model.config)

    # export onnx model
    export(
        feature_extractor, model, onnx_config,
        onnx_config.default_onnx_opset, save_path
    )

if __name__ == "__main__":
    main()