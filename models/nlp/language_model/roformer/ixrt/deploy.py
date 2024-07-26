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

import onnx
import argparse

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    model = onnx.load(args.model_path)
    for input in model.graph.input:
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == input.name:
                    node.input[i] =name.replace(':',"")
        input.name=input.name.replace(':',"")# 保存修改后的模型
    onnx.save(model, args.output_path)