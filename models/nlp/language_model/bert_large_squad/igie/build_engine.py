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
import tvm
import h5py
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
                        required=True,
                        nargs='+', 
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

    if args.precision == "int8":
        if not os.path.exists(args.engine_path):
            os.makedirs(args.engine_path)

        batchsize, seq_len = input_dict["input_ids"]
        def proc_func(values):
            residual_out = values["finaly_out"]
            with h5py.File(args.model_path) as f:
                qa_weights = relay.frontend.get_conf_by_name('src_embedding/qa_weights', f)
                qa_bias = relay.frontend.get_conf_by_name('src_embedding/qa_bias', f)
            weight = relay.const(qa_weights, dtype="float32")
            bias = relay.const(qa_bias, dtype="float32")
            residual_out = relay.nn.dense(residual_out, weight)
            residual_out = relay.reshape(residual_out, (batchsize, seq_len, -1))
            return residual_out
        
        mod = relay.frontend.bert_from_hdf5(args.model_path, batchsize, seq_len, proc_func=proc_func)
        mod.save_module(dir=args.engine_path, prefix=f"bert_large_squad_int8_b{batchsize}_seq{seq_len}")
    else:   
        mod, params = import_model_to_igie(args.model_path, input_dict, backend="igie")

        # build engine
        lib = tvm.relay.build(mod, target=target, params=params, precision=args.precision)

        # export engine
        lib.export_library(args.engine_path)



if __name__ == "__main__":
    main()