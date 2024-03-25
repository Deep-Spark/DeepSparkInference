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
import numpy as np
from tvm import relay

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

    if not os.path.exists(args.engine_path):
        os.makedirs(args.engine_path)

    batchsize, seq_len = input_dict["input_ids"]

    def proc_func(values):
        residual_out = values["finaly_out"]
        mask = values["mask"]
        with h5py.File(args.model_path) as f:
            fc_weights = relay.frontend.get_conf_by_name('src_embedding/fc_weights', f)
            fc_bias = relay.frontend.get_conf_by_name('src_embedding/fc_bias', f)
            start_transitions = relay.const(np.array(f['src_embedding']['start_transitions']), dtype="float32")
            transitions = relay.const(np.array(f['src_embedding']['transitions']), dtype="float32")
            end_transitions = relay.const(np.array(f['src_embedding']['end_transitions']), dtype="float32")
        weight = relay.const(fc_weights, dtype="float32")
        bias = relay.const(fc_bias, dtype="float32")
        residual_out = relay.nn.dense(residual_out, weight) + bias
        residual_out = relay.reshape(residual_out, (batchsize, seq_len, -1))
        residual_out = relay.nn.viterbi_decode(residual_out, mask, start_transitions, transitions, end_transitions)
        return residual_out
    
    mod = relay.frontend.bert_from_hdf5(args.model_path, batchsize, seq_len, proc_func=proc_func)
    mod.save_module(dir=args.engine_path, prefix=f"bert_base_ner_int8_b{batchsize}_seq{seq_len}")


if __name__ == "__main__":
    main()