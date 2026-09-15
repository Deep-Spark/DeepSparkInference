# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import inspect

import torch

from backbones import get_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight",
                        type=str,
                        required=True,
                        help="pytorch model weight.")

    parser.add_argument("--batch",
                        type=int,
                        required=True,
                        help="batchsize of the model.")

    parser.add_argument("--network",
                        type=str,
                        default="r50",
                        help="backbone network: r18/r34/r50/r100.")

    parser.add_argument("--output",
                        type=str,
                        default="arcface_r50.onnx",
                        help="export onnx model path.")

    parser.add_argument("--imgsz",
                        type=int,
                        default=112,
                        help="input size h=w.")

    args = parser.parse_args()
    return args


def load_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return state


def main():
    args = parse_args()

    backbone = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    state_dict = torch.load(args.weight, map_location="cpu")
    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()

    dummy_input = torch.randn(args.batch, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    torch.onnx.export(
        backbone, dummy_input, args.output,
        input_names=["input"],
        output_names=["fc1"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False
    )
    
    print(f"Export onnx model successfully! {args.output}")

if __name__ == "__main__":
    main()
