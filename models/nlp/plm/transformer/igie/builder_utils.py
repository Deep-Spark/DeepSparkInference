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
#


import onnx
import numpy as np
import tensorrt as trt
import json
import torch

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def reformat_weight_name(name):
    
    i = name[1]
    #emb
    if name.find("pos_emb_weight") !=-1:
        return name 
    
    if name.find("token_emb_weight") !=-1:
        return name 
    
    if name.find("enc_token_emb_weight") !=-1:
        return name 
    
    if name.find("enc_pos_emb_weight") !=-1:
        return name 
#################################################################
#enccoder layer weights
    #self atten to_q、to_q、to_v compute together
    if name.find("enc_self_attn_qkv_weight") !=-1:
        return f"encoder.layers.{i}.self_attn.qkv_proj.weight"
    if name.find("enc_self_attn_qkv_bias") !=-1:
        return f"encoder.layers.{i}.self_attn.qkv_proj.bias"
    
    
    
    if name.find("enc_self_attn_out_proj_weight") !=-1:
        return f"encoder.layers.{i}.self_attn.out_proj.weight"
    if name.find("enc_self_attn_out_proj_bias") !=-1:
        return f"encoder.layers.{i}.self_attn.out_proj.bias"
    
    
    if name.find("enc_self_attn_ln_weight") !=-1:
        return f"encoder.layers.{i}.self_attn_layer_norm.weight"
    if name.find("enc_self_attn_ln_bias") !=-1:
        return f"encoder.layers.{i}.self_attn_layer_norm.bias"
    
        #ffn
    if name.find("enc_ff1_weight") !=-1:
        return f'encoder.layers.{i}.fc1.weight'
    if name.find("enc_ff1_bias") !=-1:
        return f'encoder.layers.{i}.fc1.bias'

    if name.find("enc_ff2_weight") !=-1:
        return f'encoder.layers.{i}.fc2.weight'
    if name.find("enc_ff2_bias") !=-1:
        return f'encoder.layers.{i}.fc2.bias'
    
    
        #layernorm
    if name.find("enc_final_ln_weight") !=-1:
        return f"encoder.layers.{i}.final_layer_norm.weight"
    if name.find("enc_final_ln_bias") !=-1:
        return f"encoder.layers.{i}.final_layer_norm.bias"
    
    
    
####################################################################
#Decoder layer self attention  weights

    #self attention
    
    #self atten to_q、to_q、to_v compute together
    if name.find("self_attn_qkv_proj_weight") !=-1:
        return f"decoder.layers.{i}.self_attn.qkv_proj.weight"
    if name.find("self_attn_qkv_proj_bias") !=-1:
        return f"decoder.layers.{i}.self_attn.qkv_proj.bias"
    
    
    #self attention proj out
    if name.find("self_attn_out_proj_weight") !=-1:
        return f"decoder.layers.{i}.self_attn.out_proj.weight"
    if name.find("self_attn_out_proj_bias") !=-1:
        return f"decoder.layers.{i}.self_attn.out_proj.bias"
    
    #layernorm
    if name.find("self_attn_ln_weight") !=-1:
        return f"decoder.layers.{i}.self_attn_layer_norm.weight"
    if name.find("self_attn_ln_bias") !=-1:
        return f"decoder.layers.{i}.self_attn_layer_norm.bias"
    
########################################################################    
########################################################################
#Decoder layer cross attention  weights 

    #self atten to_q、to_q、to_v compute split
    #to q
    if name.find("enc_attn_q_proj_weight") !=-1:
        return f'decoder.layers.{i}.encoder_attn.q_proj.weight'
    if name.find("enc_attn_q_proj_bias") !=-1:
        return f'decoder.layers.{i}.encoder_attn.q_proj.bias'
    
    #to_kv split affter
    if name.find("enc_attn_kv_proj_weight") !=-1:
        return f'decoder.layers.{i}.encoder_attn.kv_proj.weight'
    if name.find("enc_attn_kv_proj_bias") !=-1:
        return f'decoder.layers.{i}.encoder_attn.kv_proj.bias' 
    
    if name.find("enc_attn_out_proj_weight") !=-1:
        return f'decoder.layers.{i}.encoder_attn.out_proj.weight' 
    if name.find("enc_attn_out_proj_bias") !=-1:
        return f'decoder.layers.{i}.encoder_attn.out_proj.bias' 
       
    #layernorm
    if name.find("enc_attn_ln_weight") !=-1:
        return f'decoder.layers.{i}.encoder_attn_layer_norm.weight'
    if name.find("enc_attn_ln_bias") !=-1:
        return f'decoder.layers.{i}.encoder_attn_layer_norm.bias'
########################################################################    
    #ffn
    if name.find("ff1_weight") !=-1:
        return f'decoder.layers.{i}.fc1.weight'
    if name.find("ff1_bias") !=-1:
        return f'decoder.layers.{i}.fc1.bias'

    if name.find("ff2_weight") !=-1:
        return f'decoder.layers.{i}.fc2.weight'
    if name.find("ff2_bias") !=-1:
        return f'decoder.layers.{i}.fc2.bias'

    #layernorm
    if name.find("final_ln_weight") !=-1:
        return f"decoder.layers.{i}.final_layer_norm.weight"
    if name.find("final_ln_bias") !=-1:
        return f"decoder.layers.{i}.final_layer_norm.bias"
    
#############################################################    
    if name.find("linear_weight") !=-1:
        return f"decoder.output_projection.weight"
    
    
    
    else:
        return None
    
def get_onnx_weight_dict(tensor_dict, config):
    N = config.num_attention_heads
    H = config.head_size
    hidden_size = config.hidden_size

    weights_dict = dict()
    
    for name , tensor in tensor_dict.items():
    
        update_name  = reformat_weight_name(name)
        if update_name is None:
            continue
        if update_name.find("encoder_attn.kv_proj.bias") !=-1:
            k_bias = tensor[:1024]
            v_bias = tensor[1024:]
            temp_bias_name = update_name.replace("encoder_attn.kv_proj.bias","")
            k_bias_name = temp_bias_name+ "encoder_attn.k_proj.bias"
            v_bias_name = temp_bias_name+ "encoder_attn.v_proj.bias"
            weights_dict[k_bias_name] = np.ascontiguousarray(k_bias).flatten().astype(np.float32)
            weights_dict[v_bias_name] = np.ascontiguousarray(v_bias).flatten().astype(np.float32)


        elif update_name.find("encoder_attn.kv_proj.weight")!=-1:
            k_weight = tensor[:1024]
            v_weight = tensor[1024:]            
            temp_weight_name = update_name.replace("encoder_attn.kv_proj.weight","")
            k_weight_name = temp_weight_name+"encoder_attn.k_proj.weight"
            v_weight_name = temp_weight_name+"encoder_attn.v_proj.weight"
            weights_dict[k_weight_name] = np.ascontiguousarray(k_weight).flatten().astype(np.float32)
            weights_dict[v_weight_name] = np.ascontiguousarray(v_weight).flatten().astype(np.float32)
            
        if update_name.find("self_attn.qkv_proj.bias") !=-1 and update_name.find("decoder.layers") !=-1:
            temp_bias_name = update_name.replace("self_attn.qkv_proj.bias","")
            qkv_bias_name = temp_bias_name+ "self_attn.qkv_proj.bias"                      
            weights_dict[qkv_bias_name] = np.ascontiguousarray(tensor).flatten().astype(np.float32)
              
        elif update_name.find("self_attn.qkv_proj.weight") !=-1 and update_name.find("decoder.layers") !=-1:
            temp_weight_name = update_name.replace("self_attn.qkv_proj.weight","")
            qkv_weight_name = temp_weight_name+"self_attn.qkv_proj.weight"            
            weights_dict[qkv_weight_name] = np.ascontiguousarray(tensor).flatten().astype(np.float32)
                        
        else:
            flat_tensor = np.ascontiguousarray(tensor).flatten().astype(np.float32)
            weights_dict[update_name] = flat_tensor

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

def load_onnx_weights_and_quant(path, config):
    """
    Load the weights from the onnx checkpoint
    """
    model = onnx.load(path)
    weights = model.graph.initializer
    tensor_dict = dict((w.name, np.frombuffer(w.raw_data, np.float16).reshape(w.dims))
                       for w in weights)
    return get_onnx_weight_dict(tensor_dict, config)

def load_pytorch_weights_and_quant(path, config):
    """
    Load the weights from the pytorch checkpoint
    """
    state_dict = torch.load(path, map_location='cpu')["model"]
    tensor_dict = {onnx_to_trt_name(name):val.numpy() for name, val in state_dict.items()}
    return get_onnx_weight_dict(tensor_dict, config)

class transformerBaseConfig:
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
    config_path = './wmt14_en_de/transformer_config.json'
    onnx_model_path = './wmt14_en_de/transformer.onnx'
    weight_save_path = "./wmt14_en_de/transformer.wts"
    config = config = transformerBaseConfig(config_path, True)
    weights_dict = load_onnx_weights_and_quant(onnx_model_path, config)
    
    for tensor_name, tensor in weights_dict.items():
        print(tensor_name,":",tensor.shape)
    
    
    
    # f = open(weight_save_path, "w")
    # num = 0
    # for key, value in weights_dict.items():
    #     if key.find('_amax') == -1:
    #         num += 1
    
    # f.write('{}\n'.format(num))
    # for key, value in weights_dict.items():
    #     print('key: ', key)
    #     if key.find('_amax') != -1:
    #         continue
    #     f.write("{} {}".format(key, len(value)))
    #     print(len(value))
    #     for v in value:
    #         f.write(" ")
    #         f.write(struct.pack('>f', float(v)).hex())
        # f.write("\n")
