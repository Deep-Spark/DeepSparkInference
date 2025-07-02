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

ixrt_name_map = {
    "bert.embeddings.LayerNorm.bias": "bert_embeddings_layernorm_beta",
    "bert.embeddings.LayerNorm.weight" : "bert_embeddings_layernorm_gamma",
    "bert.embeddings.word_embeddings.weight" : "bert_embeddings_word_embeddings",
    "bert.embeddings.token_type_embeddings.weight" : "bert_embeddings_token_type_embeddings",
    "bert.embeddings.position_embeddings.weight" : "bert_embeddings_position_embeddings",
    "qa_outputs.weight" : "cls_squad_output_weights",
    "qa_outputs.bias" : "cls_squad_output_bias"
}

ixrt_atten_name_map = {
    "bert.encoder.layer.{}.self_attn.qkv_proj.weight" : "l{}_attention_self_qkv_kernel",
    "bert.encoder.layer.{}.self_attn.qkv_proj.bias" : "l{}_attention_self_qkv_bias",
    "bert.encoder.layer.{}.self_attn.out_proj.bias" : "l{}_attention_output_dense_bias",
    "bert.encoder.layer.{}.self_attn.out_proj.weight" : "l{}_attention_output_dense_kernel",
    "bert.encoder.layer.{}.fc1.weight" : "l{}_intermediate_dense_kernel",
    "bert.encoder.layer.{}.fc1.bias" : "l{}_intermediate_dense_bias",
    "bert.encoder.layer.{}.fc2.weight" : "l{}_output_dense_kernel",
    "bert.encoder.layer.{}.fc2.bias" : "l{}_output_dense_bias", 
    "bert.encoder.layer.{}.self_attn_layer_norm.weight" : "l{}_attention_output_layernorm_gamma",
    "bert.encoder.layer.{}.self_attn_layer_norm.bias" : "l{}_attention_output_layernorm_beta",
    "bert.encoder.layer.{}.final_layer_norm.weight" : "l{}_output_layernorm_gamma",
    "bert.encoder.layer.{}.final_layer_norm.bias" : "l{}_output_layernorm_beta",
    "bert.encoder.layer.{}.self_attn.qkv_proj.weight_quant.clip.clip_value_max" : "l{}_attention_self_qkv_wei_amax",
    "bert.encoder.layer.{}.self_attn.qkv_proj.input_quant.clip.clip_value_max" : "l{}_attention_self_qkv_in_amax",
    "bert.encoder.layer.{}.self_attn.qkv_proj.output_quant.clip.clip_value_max" : "l{}_attention_self_qkv_out_amax",
    "bert.encoder.layer.{}.self_attn.attention_quant.clip.clip_value_max" : "l{}_attention_arrange_qkv_amax",
    "bert.encoder.layer.{}.self_attn.softmax_in_quant.clip.clip_value_max" : "l{}_attention_softmax_in_amax",
    "bert.encoder.layer.{}.self_attn.atten_score_out_quant.clip.clip_value_max" : "l{}_attention_softmax_out_amax",
    "bert.encoder.layer.{}.self_attn.out_proj.input_quant.clip.clip_value_max" : "l{}_attention_output_dense_in_amax",
    "bert.encoder.layer.{}.self_attn.out_proj.output_quant.clip.clip_value_max" : "l{}_attention_output_dense_out_amax",
    "bert.encoder.layer.{}.self_attn.out_proj.weight_quant.clip.clip_value_max" : "l{}_attention_output_dense_wei_amax",
    "bert.encoder.layer.{}.fc1.input_quant.clip.clip_value_max" : "l{}_intermediate_dense_in_amax",
    "bert.encoder.layer.{}.fc1.output_quant.clip.clip_value_max" : "l{}_intermediate_dense_out_amax",
    "bert.encoder.layer.{}.fc1.weight_quant.clip.clip_value_max" : "l{}_intermediate_dense_wei_amax",
    "bert.encoder.layer.{}.fc2.input_quant.clip.clip_value_max" : "l{}_output_dense_in_amax",
    "bert.encoder.layer.{}.fc2_out_quant.clip.clip_value_max" : "l{}_output_dense_out_amax",
    "bert.encoder.layer.{}.fc2.weight_quant.clip.clip_value_max" : "l{}_output_dense_wei_amax"
}

def get_weight_dict(tensor_dict, config):
    N = config.num_attention_heads
    H = config.head_size
    hidden_size = config.hidden_size

    weights_dict = dict()
    for outname, tensor in tensor_dict.items():
        if outname.find("_amax") != -1:
            weights_dict[outname] = tensor.item()
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

            weights_dict[prefix + WQKV] = Wqkv.flatten()
            weights_dict[prefix + BQKV] = Bqkv.flatten()
        elif outname.find(BK) != -1 or outname.find(BV) != -1 or outname.find(WQ) != -1 or outname.find(WK) != -1 or outname.find(WV) != -1:
            pass
        else:
            flat_tensor = np.ascontiguousarray(tensor).flatten()
            weights_dict[outname] = flat_tensor

    return weights_dict

def pytorch_to_trt_name(state_dict, num_layer):
    tensor_dict = {}
    for name in ixrt_name_map.keys():
        tensor_dict[ixrt_name_map[name]] = state_dict[name]

    for name in ixrt_atten_name_map.keys():
        for layer_id in range(num_layer):
            key_name = name.format(layer_id)
            value_name = ixrt_atten_name_map[name].format(layer_id)
            tensor_dict[value_name] = state_dict[key_name]
    return tensor_dict

def load_pytorch_weights_and_quant(path, config):
    """
    Load the weights from the pytorch checkpoint
    """
    state_dict = torch.load(path, map_location='cpu')
    tensor_dict = pytorch_to_trt_name(state_dict, config.num_hidden_layers)
    return get_weight_dict(tensor_dict, config)

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8=False, use_trt=False):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_trt = use_trt

if __name__ == '__main__':
    bert_config_path = './data/bert-large-uncased/bert_config.json'
    pytorch_model_path = './data/bert-large-uncased/bert_large_int8_qat.bin'
    weight_save_path = "./data/bert-large-uncased/bert_large_v1_1_int8.wts"
    config = BertConfig(bert_config_path, True)
    weights_dict = load_pytorch_weights_and_quant(pytorch_model_path, config)
    f = open(weight_save_path, "w")
    num = 0
    for key, value in weights_dict.items():
        if key.find('_amax') == -1:
            num += 1
    
    f.write('{}\n'.format(num))
    for key, value in weights_dict.items():
        if key.find('_amax') != -1:
            continue
        print('key: ', key)
        f.write("{} {}".format(key, len(value)))
        print(len(value))
        for v in value:
            f.write(" ")
            f.write(struct.pack('>f', float(v)).hex())
        f.write("\n")

    f.write('{}\n'.format(len(weights_dict) - num))
    for key, value in weights_dict.items():
        if key.find('_amax') == -1:
            continue
        print('key: ', key)
        print('value: ', value)
        f.write('{} '.format(key))
        f.write(struct.pack('>f', float(weights_dict[key])).hex())
        f.write('\n')
