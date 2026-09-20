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
import os
import argparse
import json
import tensorrt as trt
import time
import sys
import ctypes
import os
import numpy as np
from builder_utils import load_onnx_weights_and_quant, load_pytorch_weights_and_quant
from builder_utils import WQKV, BQKV  # Attention Keys
from builder_utils import W_AOUT, B_AOUT, W_MID, B_MID, W_LOUT, B_LOUT  # Transformer Keys
from builder_utils import SQD_W, SQD_B  # SQuAD Output Keys

trt_version = [int(n) for n in trt.__version__.split('.')]
plugin_lib_name = "libnvinfer_plugin.so" if os.getenv('USE_TRT') == 'True' else "libixrt_plugin.so"
print(plugin_lib_name)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin(TRT_LOGGER)

plg_registry = trt.get_plugin_registry()
registry_list = plg_registry.plugin_creator_list
print("registry_list: ", [registry.name + '/' + registry.plugin_version for registry in registry_list])
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic_IxRT", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic_IxRT", "1", "")

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_trt):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_trt = use_trt

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    return int(dtype)

def custom_fc(network, input_tensor, out_dims, W, B):
    """Built-in MatMul(+bias), replaces CustomFCPluginDynamic_IxRT."""
    rank = len(input_tensor.shape)
    k = int(input_tensor.shape[-1])

    if rank > 2:
        flatten = network.add_shuffle(input_tensor)
        flatten.reshape_dims = (-1, k)
        mm_input = flatten.get_output(0)
    else:
        mm_input = input_tensor

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
    if rank > 2:
        in_shape = network.add_shape(input_tensor).get_output(0)
        leading_idx = network.add_constant(
            (rank - 1,), np.arange(rank - 1, dtype=np.int32)
        ).get_output(0)
        leading = network.add_gather_v2(
            in_shape, leading_idx, mode=trt.GatherMode.DEFAULT
        )
        leading.axis = 0
        out_dim_shape = network.add_constant(
            (1,), np.array([out_dims], dtype=np.int32)
        ).get_output(0)
        new_shape = network.add_concatenation([leading.get_output(0), out_dim_shape])
        new_shape.axis = 0
        unflatten = network.add_shuffle(out_dense.get_output(0))
        unflatten.set_input(1, new_shape.get_output(0))
        out_dense = unflatten
    return out_dense

def qkv_to_ctx(network, qkv, num_heads, head_size, hidden_size, imask):
    """Built-in add_attention_v2, replaces CustomQKVToContextPluginDynamic_IxRT (fp16)."""
    reshape_qkv = network.add_shuffle(qkv)
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
    q_scaled = network.add_elementwise(
        q, scale_const.get_output(0), trt.ElementWiseOperation.PROD
    )
    attn = network.add_attention_v2(
        q_scaled.get_output(0),
        k,
        v,
        trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.NONE,
    )
    if imask is not None:
        mask_reshape = network.add_shuffle(imask)
        mask_reshape.reshape_dims = (0, 1, 1, -1)
        attn.mask = mask_reshape.get_output(0)
    ctx_transpose = network.add_shuffle(attn.get_output(0))
    ctx_transpose.first_transpose = (0, 2, 1, 3)
    ctx = network.add_shuffle(ctx_transpose.get_output(0))
    ctx.reshape_dims = (0, 0, hidden_size)
    return ctx


def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the attention layer
    """
    B, S, hidden_size = input_tensor.shape
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    mult_all = custom_fc(network, input_tensor, 3 * hidden_size, Wall, Ball)
    return qkv_to_ctx(
        network, mult_all.get_output(0), num_heads, head_size, hidden_size, imask
    )


def skipln(prefix, config, init_dict, network, input_tensor, skip, bias=None):
    """
    Add the skip layer
    """
    idims = input_tensor.shape
    hidden_size = idims[2]

    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "beta"]
    pf_beta = trt.PluginField("beta", wbeta, trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "gamma"]
    pf_gamma = trt.PluginField("gamma", wgamma, trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)

    fields = [pf_ld, pf_beta, pf_gamma, pf_type ]

    if bias is not None:
        pf_bias = trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32)
        fields.append(pf_bias)

    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer

def ffn_trt(prefix, config, init_dict, network, input_tensor):
     # FC1 + GELU (built-in), replaces CustomGeluPluginDynamic_IxRT
    B_mid = init_dict[prefix + B_MID]
    W_mid = init_dict[prefix + W_MID]
    mid_dense = network.add_fully_connected(input_tensor, config.intermediate_size, W_mid, B_mid)
    gelu_layer = network.add_activation(mid_dense.get_output(0), trt.ActivationType.GELU_TANH)
    intermediate_act = gelu_layer.get_output(0)

    B_lout = init_dict[prefix + B_LOUT]
    W_lout = init_dict[prefix + W_LOUT]
    out_dense = network.add_fully_connected(intermediate_act, config.hidden_size, W_lout, B_lout)
    B_lout = None

    out_layer = skipln(prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), input_tensor, B_lout)
    return out_layer

def ffn(prefix, config, init_dict, network, input_tensor):
    # Built-in FC1 + GELU_TANH + FC2, replaces CustomFFNPluginDynamic_IxRT
    B_mid = init_dict[prefix + B_MID]
    W_mid = init_dict[prefix + W_MID]
    B_lout = init_dict[prefix + B_LOUT]
    W_lout = init_dict[prefix + W_LOUT]

    mid_dense = custom_fc(network, input_tensor, config.intermediate_size, W_mid, B_mid)
    gelu_layer = network.add_activation(mid_dense.get_output(0), trt.ActivationType.GELU_TANH)
    out_dense = custom_fc(network, gelu_layer.get_output(0), config.hidden_size, W_lout, None)
    out_layer = skipln(prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), input_tensor, B_lout)
    return out_layer

def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    hidden_size = idims[2]

    context_transposed = attention_layer_opt(prefix + "attention_", config, init_dict, network, input_tensor, imask)
    attention_heads = context_transposed.get_output(0)
    
    # FC0
    B_aout = init_dict[prefix + B_AOUT]
    W_aout = init_dict[prefix + W_AOUT]
    attention_out_fc = custom_fc(network, attention_heads, hidden_size, W_aout, B_aout)
    B_aout = None     
    
    skiplayer = skipln(prefix + "attention_output_layernorm_",config, init_dict, network, attention_out_fc.get_output(0), input_tensor, B_aout)
    attention_ln = skiplayer.get_output(0)

    if config.use_trt:
        ffn_layer = ffn_trt(prefix, config, init_dict, network, attention_ln)
    else:
        ffn_layer = ffn(prefix, config, init_dict, network, attention_ln)
    return ffn_layer

def bert_model(config, init_dict, network, input_tensor, input_mask):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)   
        out_layer = transformer_layer_opt(ss, config,  init_dict, network, prev_input, input_mask)
        prev_input = out_layer.get_output(0)
    return prev_input

def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    B, S, hidden_size = idims

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    dense = custom_fc(network, input_tensor, 2, W_out, B_out)

    if config.use_trt:
        OUT = network.add_shuffle(dense.get_output(0))
        OUT.second_transpose = (1, 0, 2)
        return OUT
    return dense

def emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_lengths, batch_sizes):
    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1 if len(batch_sizes) > 1 else batch_sizes[0], -1 if len(sequence_lengths) > 1 else sequence_lengths[0]))
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1 if len(batch_sizes) > 1 else batch_sizes[0], -1 if len(sequence_lengths) > 1 else sequence_lengths[0]))
    input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(-1 if len(batch_sizes) > 1 else batch_sizes[0], -1 if len(sequence_lengths) > 1 else sequence_lengths[0]))

    if len(sequence_lengths) > 1:
        profile = builder.create_optimization_profile()
        min_shape = (batch_sizes[0], sequence_lengths[0])
        opt_shape = (batch_sizes[1], sequence_lengths[1])
        max_shape = (batch_sizes[2], sequence_lengths[2])
        assert(sequence_lengths[0] <= sequence_lengths[1] and sequence_lengths[1] <= sequence_lengths[2])
        
        print('set dynamic shape -> ', min_shape, opt_shape, max_shape)
        profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
        profile.set_shape("segment_ids", min_shape, opt_shape, max_shape)
        profile.set_shape("input_mask", min_shape, opt_shape, max_shape)
        builder_config.add_optimization_profile(profile)

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"], trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"], trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"], trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"], trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"], trt.PluginFieldType.FLOAT32)

    output_fp16 = trt.PluginField("output_fp16", np.array([1 if config.use_fp16 else 0]).astype(np.int32), trt.PluginFieldType.INT32)
    mha_type = trt.PluginField("mha_type_id", np.array([get_mha_dtype(config)], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16, mha_type])
    fn = emln_plg_creator.create_plugin("embeddings", pfc)

    if config.use_trt:
        input_ids = network.add_shuffle(input_ids)
        input_ids.second_transpose = (1, 0)
        segment_ids = network.add_shuffle(segment_ids)
        segment_ids.second_transpose = (1, 0)
        input_mask = network.add_shuffle(input_mask)
        input_mask.second_transpose = (1, 0)
        inputs = [input_ids.get_output(0), segment_ids.get_output(0), input_mask.get_output(0)]
    else:
        inputs = [input_ids, segment_ids, input_mask]
    emb_layer = network.add_plugin_v2(inputs, fn)
    return emb_layer

def build_engine(batch_sizes, sequence_lengths, config, weights_dict):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    with builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        # Create the network
        emb_layer = emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_lengths, batch_sizes)
        embeddings = emb_layer.get_output(0)
        mask_idx = emb_layer.get_output(1)
        
        bert_out = bert_model(config, weights_dict, network, embeddings, mask_idx)

        squad_logits = squad_output("cls_", config, weights_dict, network, bert_out)
        squad_logits_out = squad_logits.get_output(0)

        network.mark_output(squad_logits_out)

        build_start_time = time.time()
        plan = builder.build_serialized_network(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        return plan

def str2bool(v):
    return v.lower() in ('yes', 'true')    

def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-z", "--use_trt", type=str2bool, default=False, help = "Whether to use tensorRT or IxRT")
    parser.add_argument("-x", "--onnx", required=False, help="The ONNX model file path.")
    parser.add_argument("-pt", "--pytorch", required=False, help="The PyTorch checkpoint file path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--batch-size", nargs='+', help="Batch size(s) to optimize for. The engine will be usable with any batch size below this, but may not be optimal for smaller sizes. Can be specified multiple times to optimize for more than one batch size.", type=int)
    parser.add_argument("-s", "--sequence-length", nargs='+', help="Sequence length of the BERT model", type=int)
    parser.add_argument("-c", "--config-dir", required=True,
                        help="The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-j", "--squad-json", default="squad/dev-v1.1.json", help="squad json dataset used for int8 calibration", required=False)
    parser.add_argument("-v", "--vocab-file", default="./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt", help="Path to file containing entire understandable vocab", required=False)
    parser.add_argument("--verbose", action="store_true", help="Turn on verbose logger and set profiling verbosity to DETAILED", required=False)

    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]
    args.sequence_length = args.sequence_length or [128]

    if len(args.sequence_length) not in [1, 3]:
        print("Error: You must provide <args.sequence_length> either one or three integers.")
        sys.exit(1)

    if len(args.batch_size) not in [1, 3]:
        print("Error: You must provide <args.batch_size> either one or three integers.")
        sys.exit(1)

    if args.verbose:
        TRT_LOGGER.min_severity = TRT_LOGGER.VERBOSE

    bert_config_path = args.config_dir
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(bert_config_path))

    config = BertConfig(bert_config_path, args.fp16, args.use_trt)

    if args.onnx != None:
        weights_dict = load_onnx_weights_and_quant(args.onnx, config)
    elif args.pytorch != None:
        weights_dict = load_pytorch_weights_and_quant(args.pytorch, config)
    else:
        raise RuntimeError("You need either specify TF checkpoint using option --ckpt or ONNX using option --onnx to build TRT BERT model.")

    with build_engine(args.batch_size, args.sequence_length, config, weights_dict) as serialized_engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

if __name__ == "__main__":
    main()
