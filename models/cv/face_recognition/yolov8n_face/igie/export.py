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
import functools
import inspect

import torch
from ultralytics import YOLO


def enable_torchscript_onnx_export():
    orig = torch.onnx.export
    if "dynamo" not in inspect.signature(orig).parameters:
        return

    @functools.wraps(orig)
    def _export(*args, **kwargs):
        kwargs["dynamo"] = False
        return orig(*args, **kwargs)

    torch.onnx.export = _export


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
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    enable_torchscript_onnx_export()

    model = YOLO(args.weight).cpu()

    model.export(format='onnx',
                 batch=args.batch,
                 imgsz=(640, 640),
                 optimize=True,
                 simplify=True,
                 opset=13)


if __name__ == "__main__":
    main()
