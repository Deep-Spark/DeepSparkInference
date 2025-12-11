#!/usr/bin/env python3
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

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import tensorrt
import tensorrt as trt

trt_version = [int(n) for n in trt.__version__.split(".")[:3]]

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
from load_ixrt_plugin import load_ixrt_plugin

load_ixrt_plugin(
    TRT_LOGGER
)

plg_registry = trt.get_plugin_registry()

qkv2ctx_plg_creator = plg_registry.get_plugin_creator(
    "CustomQKVToContextPluginDynamic_IxRT", "1", ""
)
skln_plg_creator = plg_registry.get_plugin_creator(
    "CustomSkipLayerNormPluginDynamic_IxRT", "1", ""
)

encoder_emb_plg_creator = plg_registry.get_plugin_creator(
        "TransformerEncoderEmb_IxRT", "1"
    )
attention_plugin_creator = plg_registry.get_plugin_creator(
        "CustomQkvCrossToContext_IxRT", "1"
    )

decoder_emb_plg_creator = plg_registry.get_plugin_creator(
        "TransformerDecoderEmb_IxRT", "1"
    )

top1_plg_creator = plg_registry.get_plugin_creator(
        "CustomArgmax_IxRT", "1"
    )

ffn_plg_creator = plg_registry.get_plugin_creator("CustomFFNPluginDynamic_IxRT", "1", "")

fc_plg_creator = plg_registry.get_plugin_creator("CustomFCPluginDynamic_IxRT", "1", "")

def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    return int(dtype)




def create_split_qkv_plugin(num_head,num_dim,index):

    plugin_registry = tensorrt.get_plugin_registry()
    assert plugin_registry

    plugin_creator = plugin_registry.get_plugin_creator("SplitQKVUpdateKVCache_IxRT", "1")
    assert plugin_creator
    
    head_num_field = tensorrt.PluginField(
    "num_head",
    np.array([num_head], dtype=np.int32),
    tensorrt.PluginFieldType.INT32)
    
    head_dim_field = tensorrt.PluginField(
    "head_dim",
    np.array([num_dim], dtype=np.int32),
    tensorrt.PluginFieldType.INT32)
    
    field_collection = tensorrt.PluginFieldCollection([head_num_field,head_dim_field ])
    plugin = plugin_creator.create_plugin(f"SplitQKVUpdateKVCache_IxRT_{index}", field_collection)

    return plugin


def create_encoder_emb_plugin(
    weights_dict,
    config
):

    embed_scale_field = trt.PluginField(
        "embed_scale",
        np.array([32], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )
    hidden_size_field = trt.PluginField(
        "hidden_size",
        np.array([config.hidden_size], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    max_pos_field = trt.PluginField(
        "max_pos",
        np.array([1024], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )

    pad_idx_field = trt.PluginField(
        "pad_idx",
        np.array([1], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )

    token_w_field = trt.PluginField(
        "enc_token_emb_weight",
        weights_dict["enc_token_emb_weight"],
        trt.PluginFieldType.FLOAT32,
    )

    pos_w_field = trt.PluginField(
        "enc_pos_emb_weight",
        weights_dict["enc_pos_emb_weight"],
        trt.PluginFieldType.FLOAT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            embed_scale_field,
            hidden_size_field,
            max_pos_field,
            pad_idx_field,
            token_w_field,
            pos_w_field,
        ]
    )

    emb_plugin = encoder_emb_plg_creator.create_plugin(
        "py_TransformerEncoderEmb_ixrt", field_collection
    )

    return emb_plugin  



def custom_fc(network, input_tensor, out_dims, W, B):
    pf_out_dims = trt.PluginField("out_dims", np.array(out_dims, dtype=np.int32), trt.PluginFieldType.INT32)
    pf_type = trt.PluginField("type_id", np.array(int(trt.float16), dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField("W", W, trt.PluginFieldType.FLOAT32)
    fields = [pf_out_dims, pf_type, pf_W]
    if B is not None:
        pf_B = trt.PluginField("B", B, trt.PluginFieldType.FLOAT32)
        fields.append(pf_B)

    pfc = trt.PluginFieldCollection(fields)
    fc_plugin = fc_plg_creator.create_plugin("fcplugin", pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense          
 
 
 
def create_encoder_attention_plugin():
   plugin_registry = tensorrt.get_plugin_registry()
   assert plugin_registry
   plugin_creator = plugin_registry.get_plugin_creator(
       "CustomQkvCrossToContext_IxRT", "1"
   )
   assert plugin_creator
   type_id_field = tensorrt.PluginField(
       "type_id",
       np.array([1], dtype=np.int32),
       tensorrt.PluginFieldType.INT32,
   )
   has_mask_field = tensorrt.PluginField(
       "has_mask",
       np.array([1], dtype=np.int32),
       tensorrt.PluginFieldType.INT32,
   )
   
   mask_type_field = tensorrt.PluginField(
       "type_mask",
       np.array([3], dtype=np.int32),
       tensorrt.PluginFieldType.INT32,
   )
   
   scale_field = tensorrt.PluginField(
       "scale",
       np.array([1.0 / 8], dtype=np.float32),  # 1 / sqrt(head_num)
       tensorrt.PluginFieldType.FLOAT32,
   )
   field_collection = tensorrt.PluginFieldCollection([type_id_field, has_mask_field,mask_type_field,scale_field])
   plugin = plugin_creator.create_plugin("py_QkvCrossToContext_ixrt", field_collection)
   return plugin


           
def encoder_self_attention_layer(
    block, layer_index, config, init_dict, network, input_tensor, imask=None
):
    """
    Add the attention layer
    """

    B, S, hidden_size, _, _ = input_tensor.shape
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    self_attn_qkv_proj_weight = init_dict[
        f"{block}.layers.{layer_index}.self_attn.qkv_proj.weight"
    ]
    self_attn_qkv_proj_bias = init_dict[
        f"{block}.layers.{layer_index}.self_attn.qkv_proj.bias"
    ]

    # q_proj,k_proj,v_proj
    # to_qkv = network.add_fully_connected(
    #     input_tensor,
    #     3 * hidden_size,
    #     self_attn_qkv_proj_weight,
    #     self_attn_qkv_proj_bias,
    # )
    
    to_qkv = custom_fc(network, input_tensor, 3 * hidden_size, self_attn_qkv_proj_weight, self_attn_qkv_proj_bias)

    has_mask = imask is not None
    # QKV2CTX
    pf_type = trt.PluginField(
        "type_id",
        np.array([get_mha_dtype(config)], np.int32),
        trt.PluginFieldType.INT32,
    )
    pf_hidden_size = trt.PluginField(
        "hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32
    )
    pf_num_heads = trt.PluginField(
        "num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32
    )
    pf_has_mask = trt.PluginField(
        "has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32
    )
    pfc = trt.PluginFieldCollection(
        [pf_hidden_size, pf_num_heads, pf_has_mask, pf_type]
    )
    qkv2ctx_plug = qkv2ctx_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [to_qkv.get_output(0)]
    if has_mask:
        qkv_in.append(imask)
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)
    return qkv2ctx

def skipln(
    block, layer_index, name, config, init_dict, network, input_tensor, skip, bias=None
):
    """
    Add the skip layer
    """
    idims = input_tensor.shape
    
    # assert len(idims) == 5
    hidden_size = idims[2]

    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16

    pf_ld = trt.PluginField(
        "ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32
    )

    ln_weight = init_dict[f"{block}.layers.{layer_index}.{name}.weight"]
    pf_gamma = trt.PluginField("gamma", ln_weight, trt.PluginFieldType.FLOAT32)

    ln_bias = init_dict[f"{block}.layers.{layer_index}.{name}.bias"]
    pf_beta = trt.PluginField("beta", ln_bias, trt.PluginFieldType.FLOAT32)

    pf_type = trt.PluginField(
        "type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32
    )
    fields = [pf_ld, pf_beta, pf_gamma, pf_type]

    if bias is not None:
        pf_bias = trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32)
        fields.append(pf_bias)

    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer

def ffn(block, layer_index, config, init_dict, network, input_tensor):

    fc1_weight = init_dict[f"{block}.layers.{layer_index}.fc1.weight"]
    fc1_bias = init_dict[f"{block}.layers.{layer_index}.fc1.bias"]

    # mid_dense = network.add_fully_connected(
    #     input_tensor, config.intermediate_size, fc1_weight, fc1_bias
    # )
    # mid_dense = custom_fc(network, input_tensor, config.intermediate_size, fc1_weight, fc1_bias)
    

    # relu_inputs = mid_dense.get_output(0)
    # relu_layer = network.add_activation(relu_inputs, tensorrt.ActivationType.RELU)

    # intermediate_act = relu_layer.get_output(0)

    fc2_weight = init_dict[f"{block}.layers.{layer_index}.fc2.weight"]
    fc2_bias = init_dict[f"{block}.layers.{layer_index}.fc2.bias"]
    # out_dense = network.add_fully_connected(
    #     intermediate_act, config.hidden_size, fc2_weight, fc2_bias
    # )
    # out_dense = custom_fc(network, intermediate_act, config.hidden_size, fc2_weight, fc2_bias)
    
    
    pf_out_dim = trt.PluginField("out_dims", np.array(config.hidden_size, np.int32), trt.PluginFieldType.INT32)
    pf_type = trt.PluginField("type_id", np.array(int(trt.float16), np.int32), trt.PluginFieldType.INT32)
    pf_W1 = trt.PluginField("W1", fc1_weight, trt.PluginFieldType.FLOAT32)
    pf_B1 = trt.PluginField("B1", fc1_bias, trt.PluginFieldType.FLOAT32)
    pf_W2 = trt.PluginField("W2", fc2_weight, trt.PluginFieldType.FLOAT32)
    pf_act_type = trt.PluginField("act_type", np.array(int(4), np.int32), trt.PluginFieldType.INT32) #RELU=4
    pfc = trt.PluginFieldCollection([pf_out_dim, pf_type, pf_W1, pf_W2, pf_B1, pf_act_type])
    ffn_plug = ffn_plg_creator.create_plugin("ffn", pfc)

    ffn_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(ffn_inputs, ffn_plug)
    
    out_layer = skipln(
        block,
        layer_index,
        "final_layer_norm",
        config,
        init_dict,
        network,
        out_dense.get_output(0),
        input_tensor,
        fc2_bias
    )
    return out_layer

def transformer_encoder_layer(
    block, layer_index, config, init_dict, network, input_tensor, imask
):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    self_attention = encoder_self_attention_layer(
        block, layer_index, config, init_dict, network, input_tensor,imask
    )  # l0_enc_self_attn_qkv_weight  l0_enc_self_attn_qkv_bias
    
    # self_attention = encoder_self_attention_layer2(
    #     block, layer_index, config, init_dict, network, input_tensor,imask
    # )  


    self_attn_out_proj_weight = init_dict[
        f"{block}.layers.{layer_index}.self_attn.out_proj.weight"
    ]
    self_attn_out_proj_bias = init_dict[
        f"{block}.layers.{layer_index}.self_attn.out_proj.bias"
    ]

    # out_proj = network.add_fully_connected(
    #     self_attention.get_output(0),
    #     hidden_size,
    #     self_attn_out_proj_weight,
    #     self_attn_out_proj_bias,
    # )
    out_proj = custom_fc(network, self_attention.get_output(0), hidden_size, self_attn_out_proj_weight, self_attn_out_proj_bias)
    

    self_attention_skipln = skipln(
        block,
        layer_index,
        "self_attn_layer_norm",
        config,
        init_dict,
        network,
        out_proj.get_output(0),
        input_tensor,
    )
    attention_ln = self_attention_skipln.get_output(0)

    ffn_layer = ffn(block, layer_index, config, init_dict, network, attention_ln)

    return ffn_layer


def create_decoder_emb_plugin(weights_dict):

    plugin_registry = tensorrt.get_plugin_registry()
    assert plugin_registry
    plugin_creator = plugin_registry.get_plugin_creator(
        "TransformerDecoderEmb_IxRT", "1"
    )
    assert plugin_creator

    embed_scale_field = tensorrt.PluginField(
        "embed_scale",
        np.array([32], dtype=np.float32),
        tensorrt.PluginFieldType.FLOAT32,
    )
    embed_dim_field = tensorrt.PluginField(
        "embed_dim",
        np.array([1024], dtype=np.int32),
        tensorrt.PluginFieldType.INT32,
    )
    pad_idx_field = tensorrt.PluginField(
        "pad_idx",
        np.array([1], dtype=np.int32),
        tensorrt.PluginFieldType.INT32,
    )

    token_w = weights_dict["token_emb_weight"]
    token_w_field = tensorrt.PluginField(
        "token_emb_weight",
        token_w.astype(np.float16),
        tensorrt.PluginFieldType.FLOAT16,
    )

    pos_w = weights_dict["pos_emb_weight"]

    pos_w_field = tensorrt.PluginField(
        "pos_emb_weight",
        pos_w.astype(np.float16),
        tensorrt.PluginFieldType.FLOAT16,
    )

    field_collection = tensorrt.PluginFieldCollection(
        [
            embed_scale_field,
            embed_dim_field,
            pad_idx_field,
            token_w_field,
            pos_w_field,
        ]
    )

    plugin = plugin_creator.create_plugin(
        "py_TransformerDecoderEmb_ixrt", field_collection
    )

    return plugin


def create_decoder_self_attention_plugin():

    plugin_registry = tensorrt.get_plugin_registry()
    assert plugin_registry

    plugin_creator = plugin_registry.get_plugin_creator(
        "CustomQkvCrossToContext_IxRT", "1"
    )
    assert plugin_creator

    type_id_field = tensorrt.PluginField(
        "type_id",
        np.array([1], dtype=np.int32),
        tensorrt.PluginFieldType.INT32,
    )

    has_mask_field = tensorrt.PluginField(
        "has_mask",
        np.array([0], dtype=np.int32),
        tensorrt.PluginFieldType.INT32,
    )
    
    mask_type_field = tensorrt.PluginField(
       "type_mask",
       np.array([3], dtype=np.int32),
       tensorrt.PluginFieldType.INT32,
   )
   
    scale_field = tensorrt.PluginField(
       "scale",
       np.array([1.0 / 8], dtype=np.float32),  # 1 / sqrt(head_num)
       tensorrt.PluginFieldType.FLOAT32,
   )

    field_collection = tensorrt.PluginFieldCollection([type_id_field, has_mask_field,mask_type_field,scale_field])

    plugin = plugin_creator.create_plugin("py_QkvCrossToContext_ixrt", field_collection)

    return plugin



def create_cross_attention_plugin():

    plugin_registry = tensorrt.get_plugin_registry()
    assert plugin_registry

    plugin_creator = plugin_registry.get_plugin_creator(
        "CustomQkvCrossToContext_IxRT", "1"
    )
    assert plugin_creator

    type_id_field = tensorrt.PluginField(
        "type_id",
        np.array([1], dtype=np.int32),
        tensorrt.PluginFieldType.INT32,
    )

    has_mask_field = tensorrt.PluginField(
        "has_mask",
        np.array([1], dtype=np.int32),
        tensorrt.PluginFieldType.INT32,
    )
    
    mask_type_field = tensorrt.PluginField(
       "type_mask",
       np.array([3], dtype=np.int32),
       tensorrt.PluginFieldType.INT32,
   )
   
    scale_field = tensorrt.PluginField(
       "scale",
       np.array([1.0 / 8], dtype=np.float32),  # 1 / sqrt(head_num)
       tensorrt.PluginFieldType.FLOAT32,
   )

    field_collection = tensorrt.PluginFieldCollection([type_id_field, has_mask_field,mask_type_field,scale_field])

    plugin = plugin_creator.create_plugin("py_QkvCrossToContext_ixrt", field_collection)

    return plugin



def cross_attention_kv_cache(
    block, layer_index, config, init_dict, network, encoder_out
):

    """
    Add the cross attention layer
    """

    to_k_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.k_proj.weight"
    ]
    to_k_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.k_proj.bias"
    ]
    # to_k_layer = network.add_fully_connected(
    #     encoder_out, config.hidden_size, to_k_layer_weight, to_k_layer_bias
    # )
    to_k_layer = custom_fc(network, encoder_out, config.hidden_size, to_k_layer_weight, to_k_layer_bias)
    
    k_output = to_k_layer.get_output(0)
    k_t_layer = network.add_shuffle(k_output)
    k_t_layer.reshape_dims = trt.Dims(
        [0, -1, config.num_attention_heads, config.head_size]
    )
    k_t_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    input_k = k_t_layer.get_output(0)

    to_v_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.v_proj.weight"
    ]
    to_v_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.v_proj.bias"
    ]
    # to_v_layer = network.add_fully_connected(
    #     encoder_out, config.hidden_size, to_v_layer_weight, to_v_layer_bias
    # )
    to_v_layer = custom_fc(network, encoder_out, config.hidden_size, to_v_layer_weight, to_v_layer_bias)
    
    v_output = to_v_layer.get_output(0)
    v_t_layer = network.add_shuffle(v_output)
    v_t_layer.reshape_dims = trt.Dims(
        [0, -1, config.num_attention_heads, config.head_size]
    )
    v_t_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    input_v = v_t_layer.get_output(0)

    return input_k,input_v


def decoder_cross_attention_layer(
    block, layer_index, config, init_dict, network, input_tensor, imask, encoder_out
):

    """
    Add the cross attention layer
    """
    to_q_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.q_proj.weight"
    ]
    to_q_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.q_proj.bias"
    ]
    # to_q_layer = network.add_fully_connected(
    #     input_tensor, config.hidden_size, to_q_layer_weight, to_q_layer_bias
    # )
    
    print("input_tensor:",input_tensor.shape)
    
    to_q_layer = custom_fc(network, input_tensor, config.hidden_size, to_q_layer_weight, to_q_layer_bias)
    
    q_output = to_q_layer.get_output(0)

    q_t_layer = network.add_shuffle(q_output)
    q_t_layer.reshape_dims = trt.Dims(
        [0, -1, config.num_attention_heads, config.head_size]
    )  # reshape  [bs,sequence_len, hidden_size] -->[bs,sequence_len,num_attention_heads ,head_dim]
    q_t_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    input_q = q_t_layer.get_output(0)

    to_k_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.k_proj.weight"
    ]
    to_k_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.k_proj.bias"
    ]
    # to_k_layer = network.add_fully_connected(
    #     encoder_out, config.hidden_size, to_k_layer_weight, to_k_layer_bias
    # )
    
    to_k_layer = custom_fc(network, encoder_out, config.hidden_size, to_k_layer_weight, to_k_layer_bias)
    
    
    k_output = to_k_layer.get_output(0)
    k_t_layer = network.add_shuffle(k_output)
    k_t_layer.reshape_dims = trt.Dims(
        [0, -1, config.num_attention_heads, config.head_size]
    )
    k_t_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    input_k = k_t_layer.get_output(0)

    to_v_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.v_proj.weight"
    ]
    to_v_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.v_proj.bias"
    ]
    # to_v_layer = network.add_fully_connected(
    #     encoder_out, config.hidden_size, to_v_layer_weight, to_v_layer_bias
    # )
    
    to_v_layer = custom_fc(network, encoder_out, config.hidden_size, to_v_layer_weight, to_v_layer_bias)
    
    
    v_output = to_v_layer.get_output(0)
    v_t_layer = network.add_shuffle(v_output)
    v_t_layer.reshape_dims = trt.Dims(
        [0, -1, config.num_attention_heads, config.head_size]
    )
    v_t_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    input_v = v_t_layer.get_output(0)

    attention_plug = create_cross_attention_plugin()
    atten = network.add_plugin_v2([input_q, input_k, input_v,imask], attention_plug)
    
    scores = atten.get_output(0)
    scores_t_layer = network.add_shuffle(scores)
    scores_t_layer.first_transpose = trt.Permutation([0, 2, 1, 3])
    scores_t_layer.reshape_dims = trt.Dims([0, 0, config.num_attention_heads*config.head_size, 1, 1])

    scores_out = scores_t_layer.get_output(0)
    to_out_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.out_proj.weight"
    ]
    to_out_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.out_proj.bias"
    ]
    # to_out_layer = network.add_fully_connected(
    #     scores_out, config.hidden_size, to_out_layer_weight, to_out_layer_bias
    # )
    to_out_layer = custom_fc(network, scores_out, config.hidden_size, to_out_layer_weight, to_out_layer_bias)
    

    return to_out_layer






def decoder_cross_attention_kvcache_layer(
    block, layer_index, config, init_dict, network, input_tensor, imask, encoder_out, encoder_kv_cache_inputs
):

    """
    Add the cross attention layer
    """
    to_q_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.q_proj.weight"
    ]
    to_q_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.q_proj.bias"
    ]
    # to_q_layer = network.add_fully_connected(
    #     input_tensor, config.hidden_size, to_q_layer_weight, to_q_layer_bias
    # )
    
    to_q_layer = custom_fc(network, input_tensor, config.hidden_size, to_q_layer_weight, to_q_layer_bias)
    
    
    q_output = to_q_layer.get_output(0)

    q_t_layer = network.add_shuffle(q_output)
    q_t_layer.reshape_dims = trt.Dims(
        [0, -1, config.num_attention_heads, config.head_size]
    )  # reshape  [bs,sequence_len, hidden_size] -->[bs,sequence_len,num_attention_heads ,head_dim]
    q_t_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    input_q = q_t_layer.get_output(0)
    
    
    input_k = encoder_kv_cache_inputs[f"past_key_values.{layer_index}.encoder.key"]
    input_v = encoder_kv_cache_inputs[f"past_key_values.{layer_index}.encoder.value"]

  
    attention_plug = create_cross_attention_plugin()
    atten = network.add_plugin_v2([input_q, input_k, input_v,imask], attention_plug)
    
    # atten = attention2(network,input_q, input_k, input_v)

    scores = atten.get_output(0)
    scores_t_layer = network.add_shuffle(scores)
    scores_t_layer.first_transpose = trt.Permutation([0, 2, 1, 3])
    scores_t_layer.reshape_dims = trt.Dims([0, 0, config.num_attention_heads*config.head_size, 1, 1])

    scores_out = scores_t_layer.get_output(0)
    to_out_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.out_proj.weight"
    ]
    to_out_layer_bias = init_dict[
        f"{block}.layers.{layer_index}.encoder_attn.out_proj.bias"
    ]
    # to_out_layer = network.add_fully_connected(
    #     scores_out, config.hidden_size, to_out_layer_weight, to_out_layer_bias
    # )
    
    to_out_layer = custom_fc(network, scores_out, config.hidden_size, to_out_layer_weight, to_out_layer_bias)
    

    return to_out_layer


def decoder_self_attention_layer(
    block,
    layer_index,
    config,
    init_dict,
    network,
    input_tensor,
    imask,
    encoder_out,
    steps,
    kv_cache_inputs,
    kv_cache_outputs
):

    """
    Add the cross attention layer
    """
    to_qkv_layer_weight = init_dict[
        f"{block}.layers.{layer_index}.self_attn.qkv_proj.weight"
    ]
    to_qkv_layer_bias = init_dict[f"{block}.layers.{layer_index}.self_attn.qkv_proj.bias"]

    to_qkv_layer = custom_fc(network, input_tensor, 3*config.hidden_size, to_qkv_layer_weight, to_qkv_layer_bias)
        
    linear_qkv_output = to_qkv_layer.get_output(0)
    reshape_qkv_layer = network.add_shuffle(linear_qkv_output)
    reshape_qkv_layer.reshape_dims = trt.Dims(
        [0, 0, 0]
    )
    
    split_qkv_plugin = create_split_qkv_plugin(config.num_attention_heads,config.head_size,layer_index)
    split_qkv_layers = network.add_plugin_v2([reshape_qkv_layer.get_output(0), kv_cache_inputs[f"past_key_values.{layer_index}.decoder.key"],
                                                kv_cache_inputs[f"past_key_values.{layer_index}.decoder.value"]], split_qkv_plugin)
        
    input_q = split_qkv_layers.get_output(0)
    present_key = split_qkv_layers.get_output(1)
    present_value = split_qkv_layers.get_output(2)
    
    attention_plug = create_decoder_self_attention_plugin()
    atten = network.add_plugin_v2([input_q, present_key, present_value], attention_plug)
    
    scores = atten.get_output(0)
    
    scores_t_layer = network.add_shuffle(scores)
    scores_t_layer.first_transpose = trt.Permutation([0, 2, 1, 3])
    scores_t_layer.reshape_dims = trt.Dims([0, 0, config.num_attention_heads*config.head_size, 1, 1])
    
    
    kv_cache_outputs[f"present_key_values.{layer_index}.decoder.key"] = present_key
    kv_cache_outputs[f"present_key_values.{layer_index}.decoder.value"] = present_value
    

    return scores_t_layer


def transformer_decoder_layer(
    block,
    layer_index,
    config,
    init_dict,
    network,
    input_tensor,
    imask,
    encoder_out,
    steps,
    kv_cache_inputs,
    kv_cache_outputs,
    encoder_kv_cache_inputs
):
    

    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]
    self_attention = decoder_self_attention_layer(
        block,
        layer_index,
        config,
        init_dict,
        network,
        input_tensor,
        imask,
        encoder_out,
        steps,
        kv_cache_inputs,
        kv_cache_outputs
    )
    self_attn_out_proj_weight = init_dict[
        f"{block}.layers.{layer_index}.self_attn.out_proj.weight"
    ]
    self_attn_out_proj_bias = init_dict[
        f"{block}.layers.{layer_index}.self_attn.out_proj.bias"
    ]
    
    # out_proj = network.add_fully_connected(
    #     self_attention.get_output(0),
    #     hidden_size,
    #     self_attn_out_proj_weight,
    #     self_attn_out_proj_bias,
    # )
    
    out_proj = custom_fc(network, self_attention.get_output(0), hidden_size, self_attn_out_proj_weight, self_attn_out_proj_bias)
    
    self_attention_skipln = skipln(
        block,
        layer_index,
        "self_attn_layer_norm",
        config,
        init_dict,
        network,
        out_proj.get_output(0),
        input_tensor,
    )

    query = self_attention_skipln.get_output(0)
    # cross_attention = decoder_cross_attention_layer(
    #     block, layer_index, config, init_dict, network, query, imask, encoder_out
    # )
    
    cross_attention = decoder_cross_attention_kvcache_layer(
        block, layer_index, config, init_dict, network, query, imask, encoder_out,encoder_kv_cache_inputs
    )
    crosss_attention_skipln = skipln(
        block,
        layer_index,
        "encoder_attn_layer_norm",
        config,
        init_dict,
        network,
        cross_attention.get_output(0),
        query,
    )
    attention_ln = crosss_attention_skipln.get_output(0)

    ffn_layer = ffn(block, layer_index, config, init_dict, network, attention_ln)

    return ffn_layer




def create_top1_plugin():
    pad_idx_field = trt.PluginField(
        "pad_idx",
        np.array([1], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )

    field_collection = trt.PluginFieldCollection(
        [pad_idx_field]
    )

    plugin = top1_plg_creator.create_plugin(
        "argmax", field_collection
    )

    return plugin  

