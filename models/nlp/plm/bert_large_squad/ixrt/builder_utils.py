# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
import numpy as np
import tensorrt as trt
import json
import struct
import torch

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

"""
Attentions Keys
"""
WQ = "self_query_kernel"
BQ = "self_query_bias"
WK = "self_key_kernel"
BK = "self_key_bias"
WV = "self_value_kernel"
BV = "self_value_bias"
WQKV = "self_qkv_kernel"
BQKV = "self_qkv_bias"

"""
Transformer Keys
"""
W_AOUT = "attention_output_dense_kernel"
B_AOUT = "attention_output_dense_bias"
AOUT_LN_BETA = "attention_output_layernorm_beta"
AOUT_LN_GAMMA = "attention_output_layernorm_gamma"
W_MID = "intermediate_dense_kernel"
B_MID = "intermediate_dense_bias"
W_LOUT = "output_dense_kernel"
B_LOUT = "output_dense_bias"
LOUT_LN_BETA = "output_layernorm_beta"
LOUT_LN_GAMMA = "output_layernorm_gamma"

"""
Squad Output Keys
"""
SQD_W = "squad_output_weights"
SQD_B = "squad_output_bias"


def get_onnx_weight_dict(tensor_dict, config):
    N = config.num_attention_heads
    H = config.head_size
    hidden_size = config.hidden_size

    weights_dict = dict()
    for outname, tensor in tensor_dict.items():
        if outname.find("_amax") != -1:
            weights_dict[outname] = tensor.flatten()
        elif outname.find(BQ) != -1:
            prefix = outname[:outname.find(BQ)]

            Wqkv = np.zeros((3, hidden_size, hidden_size), np.float32)
            Bqkv = np.zeros((3, hidden_size), np.float32)

            Wqkv[0,:,:] = tensor_dict[prefix + WQ]
            Wqkv[1,:,:] = tensor_dict[prefix + WK]
            Wqkv[2,:,:] = tensor_dict[prefix + WV]
            Bqkv[0,:] = tensor
            Bqkv[1,:] = tensor_dict[prefix + BK]
            Bqkv[2,:] = tensor_dict[prefix + BV]

            if config.use_trt:
                Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)).transpose((1,0,2,3,4)))
                Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)).transpose((1,0,2)))

            weights_dict[prefix + WQKV] = Wqkv.flatten()
            weights_dict[prefix + BQKV] = Bqkv.flatten()
            weights_dict[prefix + WQKV + "_notrans"] = np.ascontiguousarray(Wqkv.T).flatten()

        elif outname.find(BK) != -1 or outname.find(BV) != -1 or outname.find(WQ) != -1 or outname.find(WK) != -1 or outname.find(WV) != -1:
            pass
        else:
            flat_tensor = np.ascontiguousarray(tensor).flatten()
            weights_dict[outname] = flat_tensor

            if outname.find("kernel") != -1 and config.use_trt:
                tensor = np.transpose(tensor)
                weights_dict[outname + "_notrans"] = np.ascontiguousarray(tensor).flatten()

    return weights_dict

def onnx_to_trt_name(onnx_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    qkv_strings = {'key', 'value', 'query', 'query_key_value'}
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]
    if toks[0] == 'bert': #embeddings or encoder
        if toks[1] == 'encoder': #transformer
            # Token conversions for sparse checkpoints
            if toks[-2] == 'dense_act':
                toks[-2] = 'dense'
            elif toks[-3] == 'dense_act':
                if toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'
                elif toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                toks[-3] = 'dense'
            elif toks[-2].startswith('matmul'):
                toks[-2] = {
                    'matmul_q_quantizer': 'qv_a_input_quantizer',
                    'matmul_k_quantizer': 'qv_b_input_quantizer',
                    'matmul_v_quantizer': 'av_b_input_quantizer',
                    'matmul_a_quantizer': 'av_a_input_quantizer',
                }[toks[-2].replace('input_', '')]

            # Token conversions for all checkpoints
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in qkv_strings) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in qkv_strings) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'

            if 'final_input_quantizer' not in toks[2]:
                ind = toks.index('layers')+1 if 'layers' in toks else 3
                toks = toks[ind:]
                toks[0] = 'l{}'.format(int(toks[0]))
        else:
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else: #embeddings: drop "_weight" suffix
                if toks[-1] == 'amax':
                    toks[-2] = 'amax'
                toks = toks[:-1]
    elif 'qa' in onnx_name:
        name = 'cls_squad_output_bias' if toks[-1] == 'bias' else 'cls_squad_output_weights'
        return name
    else:
        print("Encountered unknown case:", onnx_name)
        assert(False)
    parsed = '_'.join(toks)
    return parsed

def pt_to_trt_name(pt_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    qkv_strings = {'key', 'value', 'query', 'query_key_value'}
    pt_name = pt_name.lower()
    toks = [t.strip('_') for t in pt_name.split('.')]
    if toks[0] == 'bert': #embeddings or encoder
        if toks[1] == 'encoder': #transformer
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in qkv_strings) and toks[-1] == 'weight':
                toks[-1] = 'kernel'

            if 'final_input_quantizer' not in toks[2]:
                ind = toks.index('layers')+1 if 'layers' in toks else 3
                toks = toks[ind:]
                toks[0] = 'l{}'.format(int(toks[0]))

        else:
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else: #embeddings: drop "_weight" suffix
                toks = toks[:-1]

    elif 'qa_outputs' in pt_name: ##
        name = 'cls_squad_output_bias' if toks[-1] == 'bias' else 'cls_squad_output_weights'
        return name
    else:
        print("Encountered unknown case:", pt_name)
        assert(False)
    parsed = '_'.join(toks)
    return parsed

def load_onnx_weights_and_quant(path, config):
    """
    Load the weights from the onnx checkpoint
    """
    model = onnx.load(path)
    weights = model.graph.initializer
    # for w in weights:
    #     print(w.name, w.dims,flush=True)
    tensor_dict = dict((onnx_to_trt_name(w.name), np.frombuffer(w.raw_data, np.int8).reshape(w.dims))
                       if w.name.split('_')[-1] == 'mask' else
                       (onnx_to_trt_name(w.name), np.frombuffer(w.raw_data, np.float32).reshape(w.dims))
                       for w in weights)
    # for key in tensor_dict:
    #     print(key, tensor_dict[key].shape,flush=True)

    return get_onnx_weight_dict(tensor_dict, config)

def load_pytorch_weights_and_quant(path, config):
    """
    Load the weights from the pytorch checkpoint
    """
    state_dict = torch.load(path, map_location='cpu')
    # for name in state_dict:
    #     print(name, state_dict[name].size(),flush=True)
    tensor_dict = {pt_to_trt_name(name):val.numpy()  for name, val in state_dict.items()}
    # for key in tensor_dict:
    #     print(key, tensor_dict[key].shape,flush=True)
    return get_onnx_weight_dict(tensor_dict, config)

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8=False):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8

if __name__ == '__main__':
    bert_config_path = '../bert-large-uncased/bert_config.json'
    onnx_model_path = '../bert-large-uncased/bert_large_v1_1_fake_quant.onnx'
    weight_save_path = "../bert-large-uncased/bert_large_v1_1.wts"
    config = config = BertConfig(bert_config_path, True)
    weights_dict = load_onnx_weights_and_quant(onnx_model_path, config)
    f = open(weight_save_path, "w")
    num = 0
    for key, value in weights_dict.items():
        if key.find('_amax') == -1:
            num += 1
    
    f.write('{}\n'.format(num))
    for key, value in weights_dict.items():
        print('key: ', key)
        if key.find('_amax') != -1:
            continue
        f.write("{} {}".format(key, len(value)))
        print(len(value))
        for v in value:
            f.write(" ")
            f.write(struct.pack('>f', float(v)).hex())
        f.write("\n")
