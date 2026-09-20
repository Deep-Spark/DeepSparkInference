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

# CustomQKVToContextPluginDynamic_IxRT removed; encoder uses builtin attention.
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

# CustomFFNPluginDynamic_IxRT and CustomFCPluginDynamic_IxRT removed; replaced by builtins.

def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    return int(dtype)




def create_split_qkv_plugin(num_head,num_dim,index):
    """Deprecated: SplitQKVUpdateKVCache_IxRT replaced by split_qkv_update_kv_cache()."""
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


def split_qkv_update_kv_cache(network, qkv, past_key, past_value, num_heads, head_size):
    """Built-in Split + Concat, replaces SplitQKVUpdateKVCache_IxRT.

    qkv: [B, 1, 3H] or [B, 1, 3H, 1, 1]
    past_key / past_value: [B, num_heads, S-1, head_size]
    returns:
      q: [B, num_heads, 1, head_size]
      present_key / present_value: [B, num_heads, S, head_size]
    """
    qkv_in = qkv
    if len(qkv.shape) == 5:
        sq = network.add_shuffle(qkv)
        sq.reshape_dims = (0, 0, -1)
        qkv_in = sq.get_output(0)

    reshape_qkv = network.add_shuffle(qkv_in)
    reshape_qkv.reshape_dims = (0, 0, 3, num_heads, head_size)

    def extract(idx):
        sel = network.add_constant((), np.array(idx, dtype=np.int32))
        gathered = network.add_gather(reshape_qkv.get_output(0), sel.get_output(0), 2)
        transposed = network.add_shuffle(gathered.get_output(0))
        transposed.first_transpose = (0, 2, 1, 3)
        return transposed.get_output(0)

    q, k_new, v_new = extract(0), extract(1), extract(2)
    cat_k = network.add_concatenation([past_key, k_new])
    cat_k.axis = 2
    cat_v = network.add_concatenation([past_value, v_new])
    cat_v.axis = 2
    return q, cat_k.get_output(0), cat_v.get_output(0)


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
    """Built-in MatMul(+bias), replaces CustomFCPluginDynamic_IxRT.
    Handles 3D [B,S,H] and 5D [B,S,H,1,1] inputs (the plugin's convention).
    """
    rank = len(input_tensor.shape)

    if rank == 5:
        squeeze = network.add_shuffle(input_tensor)
        squeeze.reshape_dims = (0, 0, -1)
        in_3d = squeeze.get_output(0)
    else:
        in_3d = input_tensor

    k = int(in_3d.shape[-1])

    flatten = network.add_shuffle(in_3d)
    flatten.reshape_dims = (-1, k)
    mm_input = flatten.get_output(0)

    weight = np.ascontiguousarray(W).reshape(out_dims, k).astype(np.float16)
    weight_const = network.add_constant((out_dims, k), weight)
    out_dense = network.add_matrix_multiply(
        mm_input,
        trt.MatrixOperation.NONE,
        weight_const.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    if B is not None:
        bias = np.ascontiguousarray(B).reshape(1, out_dims).astype(np.float16)
        bias_const = network.add_constant((1, out_dims), bias)
        out_dense = network.add_elementwise(
            out_dense.get_output(0),
            bias_const.get_output(0),
            trt.ElementWiseOperation.SUM,
        )

    in_shape = network.add_shape(in_3d).get_output(0)
    leading_idx = network.add_constant((2,), np.array([0, 1], dtype=np.int32)).get_output(0)
    leading = network.add_gather_v2(in_shape, leading_idx, mode=trt.GatherMode.DEFAULT)
    leading.axis = 0
    out_dim_shape = network.add_constant((1,), np.array([out_dims], dtype=np.int32)).get_output(0)
    new_shape = network.add_concatenation([leading.get_output(0), out_dim_shape])
    new_shape.axis = 0
    unflatten = network.add_shuffle(out_dense.get_output(0))
    unflatten.set_input(1, new_shape.get_output(0))

    if rank == 5:
        unsqueeze = network.add_shuffle(unflatten.get_output(0))
        unsqueeze.reshape_dims = (0, 0, out_dims, 1, 1)
        return unsqueeze

    return unflatten          
 
 
 
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
    Add the encoder self-attention layer (builtin add_attention_v2).
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

    to_qkv = custom_fc(network, input_tensor, 3 * hidden_size, self_attn_qkv_proj_weight, self_attn_qkv_proj_bias)

    # CustomFC outputs 5D [B,S,3H,1,1]; squeeze to 3D for the builtin attention path.
    squeeze = network.add_shuffle(to_qkv.get_output(0))
    squeeze.reshape_dims = (0, 0, 3 * hidden_size)
    packed_3d = squeeze.get_output(0)

    # reshape -> gather -> transpose -> attention_v2 -> collapse (same as BERT fp16).
    reshape_qkv = network.add_shuffle(packed_3d)
    reshape_qkv.reshape_dims = (0, 0, 3, num_heads, head_size)

    def extract(idx):
        sel = network.add_constant((), np.array(idx, dtype=np.int32))
        gathered = network.add_gather(reshape_qkv.get_output(0), sel.get_output(0), 2)
        transposed = network.add_shuffle(gathered.get_output(0))
        transposed.first_transpose = (0, 2, 1, 3)
        return transposed.get_output(0)

    q = extract(0)
    k = extract(1)
    v = extract(2)

    scale = np.array(1.0 / np.sqrt(head_size), dtype=np.float16).reshape(1, 1, 1, 1)
    scale_const = network.add_constant((1, 1, 1, 1), scale)
    q_scaled = network.add_elementwise(q, scale_const.get_output(0), trt.ElementWiseOperation.PROD)

    attn = network.add_attention_v2(
        q_scaled.get_output(0), k, v,
        trt.AttentionNormalizationOp.SOFTMAX, trt.CausalMaskKind.NONE,
    )

    has_mask = imask is not None
    if has_mask:
        mask_reshape = network.add_shuffle(imask)
        mask_reshape.reshape_dims = (0, 1, 1, -1)
        attn.mask = mask_reshape.get_output(0)

    # Collapse: (B,H,S,D) -> (B,S,H,D) -> (B,S,hidden)
    ctx_transpose = network.add_shuffle(attn.get_output(0))
    ctx_transpose.first_transpose = (0, 2, 1, 3)
    ctx = network.add_shuffle(ctx_transpose.get_output(0))
    ctx.reshape_dims = (0, 0, hidden_size)

    # Unsqueeze back to 5D [B,S,H,1,1] for downstream CustomFC plugin.
    unsqueeze = network.add_shuffle(ctx.get_output(0))
    unsqueeze.reshape_dims = (0, 0, hidden_size, 1, 1)
    return unsqueeze

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
    # Built-in FC1 + RELU + FC2, replaces CustomFFNPluginDynamic_IxRT.
    fc1_weight = init_dict[f"{block}.layers.{layer_index}.fc1.weight"]
    fc1_bias = init_dict[f"{block}.layers.{layer_index}.fc1.bias"]

    fc2_weight = init_dict[f"{block}.layers.{layer_index}.fc2.weight"]
    fc2_bias = init_dict[f"{block}.layers.{layer_index}.fc2.bias"]

    mid_dense = custom_fc(network, input_tensor, config.intermediate_size, fc1_weight, fc1_bias)
    relu_layer = network.add_activation(mid_dense.get_output(0), tensorrt.ActivationType.RELU)
    out_dense = custom_fc(network, relu_layer.get_output(0), config.hidden_size, fc2_weight, None)

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

    # Built-in Split(QKV) + Concat(past, new), replaces SplitQKVUpdateKVCache_IxRT
    input_q, present_key, present_value = split_qkv_update_kv_cache(
        network,
        to_qkv_layer.get_output(0),
        kv_cache_inputs[f"past_key_values.{layer_index}.decoder.key"],
        kv_cache_inputs[f"past_key_values.{layer_index}.decoder.value"],
        config.num_attention_heads,
        config.head_size,
    )

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

