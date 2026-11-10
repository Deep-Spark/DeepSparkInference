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

from ultralytics import YOLO


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

    # Export on CPU: the exporter traces the graph, and a CoreX device is not
    # needed until the engine is built.
    model = YOLO(args.weight).cpu()

    # simplify=False keeps the graph ixRT was verified against; dynamic=False
    # fixes the batch so the engine shape matches --batch.
    model.export(
        format="onnx",
        batch=args.batch,
        imgsz=640,
        opset=13,
        simplify=False,
        dynamic=False,
        device="cpu",
    )


if __name__ == "__main__":
    main()
