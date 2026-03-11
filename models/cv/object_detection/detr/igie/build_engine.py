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

import tvm
import argparse
from tvm import relay
from tvm.relay.import_model import import_model_to_igie
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", 
                        type=str, 
                        required=True, 
                        help="original model path.")
    
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument("--precision",
                        type=str,
                        choices=["fp32", "fp16", "int8"],
                        required=True,
                        help="model inference precision.")

    
    parser.add_argument("--engine_path", 
                        type=str, 
                        required=True, 
                        help="igie export engine path.")
    
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    input_dict = {"tensor": [args.batch_size, 3, 800, 800], "mask": [args.batch_size, 800, 800]}

    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")    

    mod, params = import_model_to_igie(args.model_path, input_dict, backend="igie")

    # build engine
    lib = tvm.relay.build(mod, target=target, params=params, precision=args.precision)

    # export engine
    lib.export_library(args.engine_path)
    print("done.")


if __name__ == "__main__":
    main()