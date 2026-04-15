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
from transformers.models.vit import ViTOnnxConfig
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.utils import is_torch_available, TensorType
import torch

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

    checkpoint = "vit-base-patch16-224"
    feature_extractor = ViTImageProcessor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(checkpoint)

    save_path = Path(args.output)

    class StaticViTOnnxConfig(ViTOnnxConfig):
        default_fixed_batch = 1

        @property
        def inputs(self):
            orig_inputs = super().inputs
            orig_inputs["pixel_values"] = {
                0: "batch_size",  
                # 1: "channels",   
                # 2: "height",    
                # 3: "width",    
            }
            orig_inputs["pixel_values"] = {0: "batch_size", 1: 3, 2: 224, 3: 224}
            return orig_inputs


        def generate_dummy_inputs_for_validation(self, reference_tensors, preprocessor):
            dummy_inputs = super().generate_dummy_inputs_for_validation(reference_tensors, preprocessor)
            if is_torch_available() and isinstance(dummy_inputs.get("pixel_values"), torch.Tensor):
                if dummy_inputs["pixel_values"].shape != (1, 3, 224, 224):
                     dummy_inputs["pixel_values"] = torch.ones((1, 3, 224, 224), dtype=dummy_inputs["pixel_values"].dtype, device=dummy_inputs["pixel_values"].device)
            return dummy_inputs


    onnx_config = StaticViTOnnxConfig(model.config)

    export(
        preprocessor=feature_extractor,
        model=model,
        config=onnx_config, 
        #opset=onnx_config.default_onnx_opset,
        opset=17,
        output=save_path
    )

    print("Export onnx model successfully! ")

if __name__ == "__main__":
    main()
