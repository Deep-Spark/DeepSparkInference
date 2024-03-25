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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", 
                        type=str, 
                        required=True, 
                        help="original model path.")
    
    parser.add_argument("--engine_path", 
                        type=str, 
                        required=True, 
                        help="igie export engine path.")

    parser.add_argument("--input",
                        type=str,
                        nargs='+', 
                        required=True, 
                        help="""
                            input info of the model, format should be:
                            input_name:input_shape
                            eg: --input input:1,3,224,224.
                            """)
               
    parser.add_argument("--precision",
                        type=str,
                        choices=["fp32", "fp16", "int8"],
                        required=True,
                        help="model inference precision.")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # get input valueinfo
    input_dict = {}
    for input_info in args.input:
        input_name, input_shape = input_info.split(":")
        shape = tuple([int(s) for s in input_shape.split(",")])
        input_dict[input_name] = shape

    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")    

    mod, params = import_model_to_igie(args.model_path, input_dict, backend="igie")

    func = mod["main"]
    body = func.body
    new_body = relay.Tuple([body[0], body[1], body[2]])
    func = relay.Function(relay.analysis.free_vars(new_body), new_body)
    encoder_mod = tvm.IRModule.from_expr(func)
    encoder_mod = relay.transform.InferType()(encoder_mod)

    # build engine
    lib = tvm.relay.build(encoder_mod, target=target, params=params, precision=args.precision)

    # export engine
    lib.export_library(args.engine_path)


if __name__ == "__main__":
    main()