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

import sys
sys.path.insert(0, "yolov4")
import argparse

from yolov4.tool.darknet2onnx import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", 
                    type=str, 
                    required=True, 
                    help="darknet cfg path.")

    parser.add_argument("--weight", 
                    type=str, 
                    required=True, 
                    help="darknet weights path.")
    
    parser.add_argument("--output", 
                    type=str, 
                    required=True, 
                    help="export onnx model path.")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    transform_to_onnx(args.cfg, args.weight, -1, args.output)
    
if __name__ == "__main__":
    main()

