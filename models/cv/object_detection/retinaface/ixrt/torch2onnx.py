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

import os
import argparse
import torch
import onnx
from onnxsim import simplify


from models import mnetv1_retinaface

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="torch model path",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        help="onnx model path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_config()

    torch_pretrained = args.model
    export_onnx_file = args.onnx_model

    model = mnetv1_retinaface()
    model.eval()

    checkpoint = torch.load(torch_pretrained, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint, strict=False)

    inputs = torch.randn(32, 3, 320, 320)

    torch.onnx.export(model,
                    inputs,
                    export_onnx_file,
                    opset_version=11,
                    input_names=["input"],
                    output_names = ["bbox_out0", "cls_out0", "ldm_out0", 
                                        "bbox_out1", "cls_out1", "ldm_out1", 
                                        "bbox_out2", "cls_out2", "ldm_out2"]
                    )

    onnx_model = onnx.load(export_onnx_file)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, export_onnx_file)
    print('finished exporting onnx')