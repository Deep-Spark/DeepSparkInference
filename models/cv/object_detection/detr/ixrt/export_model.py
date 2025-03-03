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

import os
import argparse

import torch
import onnx
from onnx import shape_inference
from onnxsim import simplify


validate=True

def stat_model(onnx_file):
    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph

    op_types = list()
    for node in graph.node:
        op_types.append(node.op_type)

    print(set(op_types))

def ort_inference(onnx_file, input):
    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_file, 
                    providers=['CPUExecutionProvider'])
    in_name = ort_session.get_inputs()[0].name

    onnx_outputs = ort_session.get_outputs()
    output_names = []
    for o in onnx_outputs:
        output_names.append(o.name)
 
    input_np = input.clone().cpu().numpy()
    out = ort_session.run(output_names, 
                            input_feed={in_name: input_np}
                        )
    return out

def convert_model(onnx_file, config):
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()

    input = torch.randn([config.bsz, 3, config.img_H, config.img_W])
    out = model(input)
    torch.onnx.export(
        model,
        input,
        onnx_file,
        verbose = False,
        input_names = ["input"],
        output_names = ["pred_logits","pred_boxes"],
        opset_version = 11
    )

    onnx_model = onnx.load(onnx_file)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    onnx_model = shape_inference.infer_shapes(model_simp)

    onnx.save(onnx_model, onnx_file)
    print('finished exporting onnx')

    # stat_model(onnx_file)

    if validate:
        torch_out = model(input)["pred_logits"]
        onnx_out = ort_inference(onnx_file, input)[0]

        import numpy as np
        torch_out = torch_out.detach().numpy()
        diff = np.abs(torch_out-onnx_out).max()
        print(diff)
        #sim = cosine_similarity(torch_out.reshape(1,-1), onnx_out.reshape(1, -1))
        #print(sim[0])


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--torch_file", type=str,  help="torch model")
    parser.add_argument("--onnx_file", type=str,  help="onnx model",default="")
    parser.add_argument("--bsz", type=int, default=1, help="test batch size")
    parser.add_argument(
        "--img_H",
        type=int,
        default=800,
        help="inference size h",
    )
    parser.add_argument(
        "--img_W",
        type=int,
        default=800,
        help="inference size W",
    )


    config = parser.parse_args()
    return config

if __name__ == "__main__":

    config = parse_config()
    onnx_file = config.onnx_file
    convert_model(onnx_file, config)