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

import tensorrt as trt
from builder_utils import load_onnx_weights_and_quant
from plugin_utils import (
    TRT_LOGGER,
    create_decoder_emb_plugin,
    create_encoder_emb_plugin,
    transformer_decoder_layer,
    transformer_encoder_layer,
    cross_attention_kv_cache,
    create_top1_plugin,
    custom_fc
)

from transformer_cfg import TransformerBaseConfig


def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    return int(dtype)


def transformer_encoder(config, init_dict, network, input_tensor, input_mask):
    """
    Create the bert model
    """

    block = "encoder"
    prev_input = input_tensor
    for ss in range(config.num_hidden_layers):
        out_layer = transformer_encoder_layer(
            block, ss, config, init_dict, network, prev_input, input_mask
        )
        prev_input = out_layer.get_output(0)
    return prev_input


def transformer_decoder(
    config,
    init_dict,
    network,
    encoder_emb_out,
    input_mask,
    encoder_out,
    steps,
    kv_cache_inputs,
    kv_cache_outputs,
    encoder_kv_cache_inputs
):
    """
    Create the bert model
    """
    prev_input = encoder_emb_out
    block = "decoder"
    for ss in range(config.num_hidden_layers):
        out_layer = transformer_decoder_layer(
            block,
            ss,
            config,
            init_dict,
            network,
            prev_input,
            input_mask,
            encoder_out,
            steps,
            kv_cache_inputs,
            kv_cache_outputs,
            encoder_kv_cache_inputs
        )
        prev_input = out_layer.get_output(0)

    decoder_output_projection_weight = init_dict[f"{block}.output_projection.weight"]
    # out_proj_layer = network.add_fully_connected(
    #     prev_input, config.tgt_vocab_size, decoder_output_projection_weight
    # )  #

    out_proj_layer = custom_fc(network, prev_input, config.tgt_vocab_size, decoder_output_projection_weight, None)

    reshape_layer = network.add_shuffle(out_proj_layer.get_output(0))

    reshape_layer.reshape_dims = trt.Dims([0, -1])  # reshape [bsz,vocab_size]
    decoder_blk_out = reshape_layer.get_output(0)
    return decoder_blk_out


def build_encoder_engine(batch_sizes, sequence_lengths, config, weights_dict):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    encoder_emb_plugin = create_encoder_emb_plugin(weights_dict, config)

    builder = trt.Builder(TRT_LOGGER)
    with builder.create_network(
        explicit_batch_flag
    ) as network, builder.create_builder_config() as builder_config:

        builder_config.set_flag(trt.BuilderFlag.FP16)
        input_ids = network.add_input(
            name="src_tokens", dtype=trt.int32, shape=[-1, -1]
        )
        MIN_SHAPE = (batch_sizes[0], sequence_lengths[0])
        OPT_SHAPE = (batch_sizes[1], sequence_lengths[1])
        MAX_SHAPE = (batch_sizes[2], sequence_lengths[2])

        profile = builder.create_optimization_profile()
        profile.set_shape("src_tokens", MIN_SHAPE, OPT_SHAPE, MAX_SHAPE)
        builder_config.add_optimization_profile(profile)

        #######################{transformer Encoder emb layer}#####################
        emb_layer = network.add_plugin_v2([input_ids], encoder_emb_plugin)
        ###########################################################################
        embeddings = emb_layer.get_output(0)
        mask_idx = emb_layer.get_output(1)

        #######################{transformer Encoder  block}#####################
        
        encoder_out = transformer_encoder(
            config, weights_dict, network, embeddings, mask_idx
        )
        #######################################################################
        
        
        for layer_index in range(config.num_hidden_layers):
            block = "decoder"
            k_cache,v_cache =  cross_attention_kv_cache(block, layer_index, config, weights_dict, network, encoder_out)
            
            k_cache.name = f"past_key_values.{layer_index}.encoder.key"
            network.mark_output(k_cache)
            k_cache.dtype = trt.float16
            
            v_cache.name = f"past_key_values.{layer_index}.encoder.value"
            network.mark_output(v_cache)
            v_cache.dtype = trt.float16
            
        mask_idx.name = "mask"
        network.mark_output(mask_idx)
        mask_idx.dtype = trt.int32

        plan = builder.build_serialized_network(network, builder_config)
   
        return plan


def build_engine_decoder(batch_sizes, sequence_lengths, config, weights_dict):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    decoder_emb_plugin = create_decoder_emb_plugin(weights_dict)

    MIN_BSZ = batch_sizes[0]
    OPT_BSZ = batch_sizes[1]
    MAX_BSZ = batch_sizes[2]

    MIN_LEN = sequence_lengths[0]
    OPT_LEN = sequence_lengths[1]
    MAX_LEN = sequence_lengths[2]

    with builder.create_network(
        explicit_batch_flag
    ) as network, builder.create_builder_config() as builder_config:
        builder_config.set_flag(trt.BuilderFlag.FP16)

        ###################IxinferDecFormatEncOutput

        token_id = network.add_input(
            "token_id", dtype=trt.int32, shape=(-1, 1)
        )  # [bsz,1]
        steps = network.add_input("steps", dtype=trt.int32, shape=(1,))  # [1,1]
        mask = network.add_input(
            "mask", dtype=trt.int32, shape=(-1, -1)
        )  # [bsz,seq_len]

############################################################################################
        kv_cache_inputs = {}  # past_key_values
        kv_cache_outputs = {}  # present_key_values

        for i in range(config.num_hidden_layers):
            k_cache_name = f"past_key_values.{i}.decoder.key"
            v_cache_name = f"past_key_values.{i}.decoder.value"
            k_cache_input = network.add_input(
                k_cache_name,
                dtype=trt.float16,
                shape=(
                    -1,
                    config.num_attention_heads,
                    -1,
                    config.head_size,
                ),  # (bsz, config.num_attention_heads, steps, config.head_size)
            )
            v_cache_input = network.add_input(
                v_cache_name,
                dtype=trt.float16,
                shape=(-1, config.num_attention_heads, -1, config.head_size),
            )
            kv_cache_inputs[k_cache_name] = k_cache_input
            kv_cache_inputs[v_cache_name] = v_cache_input

        profile = builder.create_optimization_profile()
        for i in range(config.num_hidden_layers):
            k_cache_name = f"past_key_values.{i}.decoder.key"
            v_cache_name = f"past_key_values.{i}.decoder.value"
            profile.set_shape(
                k_cache_name,
                (MIN_BSZ, config.num_attention_heads, 0, config.head_size),  #0 fist step kv cache don't concat
                (OPT_BSZ, config.num_attention_heads, OPT_LEN, config.head_size),
                (MAX_BSZ, config.num_attention_heads, MAX_LEN, config.head_size),
            )
            profile.set_shape(
                v_cache_name,
                (MIN_BSZ, config.num_attention_heads, 0, config.head_size),  #0 fist step kv cache don't concat
                (OPT_BSZ, config.num_attention_heads, OPT_LEN, config.head_size),
                (MAX_BSZ, config.num_attention_heads, MAX_LEN, config.head_size),
            )
            
############################################################################################

        encoder_kv_cache_inputs = {}
        #cross attention kv cache
        for i in range(config.num_hidden_layers):
            k_cache_name = f"past_key_values.{i}.encoder.key"
            v_cache_name = f"past_key_values.{i}.encoder.value"
            k_cache_input = network.add_input(
                k_cache_name,
                dtype=trt.float16,
                shape=(
                    -1,
                    config.num_attention_heads,
                    -1,
                    config.head_size,
                ),  # (bsz, config.num_attention_heads, steps, config.head_size)
            )
            v_cache_input = network.add_input(
                v_cache_name,
                dtype=trt.float16,
                shape=(-1, config.num_attention_heads, -1, config.head_size),
            )
            encoder_kv_cache_inputs[k_cache_name] = k_cache_input
            encoder_kv_cache_inputs[v_cache_name] = v_cache_input
            
        
        for i in range(config.num_hidden_layers):
            k_cache_name = f"past_key_values.{i}.encoder.key"
            v_cache_name = f"past_key_values.{i}.encoder.value"
            profile.set_shape(
                k_cache_name,
                (MIN_BSZ, config.num_attention_heads, 1, config.head_size), 
                (OPT_BSZ, config.num_attention_heads, OPT_LEN, config.head_size),
                (MAX_BSZ, config.num_attention_heads, MAX_LEN, config.head_size),
            )
            profile.set_shape(
                v_cache_name,
                (MIN_BSZ, config.num_attention_heads, 1, config.head_size),
                (OPT_BSZ, config.num_attention_heads, OPT_LEN, config.head_size),
                (MAX_BSZ, config.num_attention_heads, MAX_LEN, config.head_size),
            )    
            
            
            
########################################################################################3###
        profile.set_shape("token_id", (MIN_BSZ, 1), (OPT_BSZ, 1), (MAX_BSZ, 1))
        profile.set_shape(
            "mask", (MIN_BSZ, MIN_LEN), (OPT_BSZ, OPT_LEN), (MAX_BSZ, MAX_LEN)
        )
        builder_config.add_optimization_profile(profile)
        
        encoder_reshape_out = None

        ############################## decodr
        encoder_emb_layer = network.add_plugin_v2([token_id, steps], decoder_emb_plugin)
        encoder_emb_out = encoder_emb_layer.get_output(0)

        ##############################

        decoder_out = transformer_decoder(
            config,
            weights_dict,
            network,
            encoder_emb_out,
            mask,
            encoder_reshape_out,
            steps,
            kv_cache_inputs,
            kv_cache_outputs,
            encoder_kv_cache_inputs
        )

        # top1_layer = network.add_topk(
        #     decoder_out, op=trt.TopKOperation.MAX, k=1, axes=2
        # )
        
        top1_plg = create_top1_plugin()
        top1_layer = network.add_plugin_v2([decoder_out], top1_plg)
        token_out = top1_layer.get_output(0)
        token_out.dtype = trt.int32
        token_out.name = "decoder_id"
        network.mark_output(token_out)

        for i in range(config.num_hidden_layers):
            k_cache_name = f"present_key_values.{i}.decoder.key"
            v_cache_name = f"present_key_values.{i}.decoder.value"
            key_out = kv_cache_outputs[k_cache_name]
            key_out.name = k_cache_name
            network.mark_output(key_out)
            key_out.dtype = trt.float16

            value_out = kv_cache_outputs[v_cache_name]
            value_out.name = v_cache_name
            network.mark_output(value_out)
            value_out.dtype = trt.float16
        plan = builder.build_serialized_network(network, builder_config)

        return plan



def main():
    parser = argparse.ArgumentParser(
        description="TensorRT Transformer Base Sample",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    
    parser.add_argument(
        "--model_dir",
        default="/inferencesamples/data/checkpoints/transformer/wmt14.en-fr.joined-dict.transformer/",
        help="The ONNX model file path.",
    )

    parser.add_argument(
        "--batch_size",
        default=[1, 64, 128], # min,opt,max
        action="append",
        help="Batch size(s) to optimize for",
        type=int,
    )
    parser.add_argument(
        "--sequence_length",
        default=[1, 64, 257], # min,opt,max
        action="append",
        help="Sequence length of the transformer model",
        type=int,
    )


    args = parser.parse_args()
    config_path = os.path.join(args.model_dir, "transformer_config.json")
    config = TransformerBaseConfig(config_path)
    onnx_path = os.path.join(args.model_dir, "transformer.onnx")
    weights_dict = load_onnx_weights_and_quant(onnx_path, config)
    
    
    encoder_path = os.path.join(args.model_dir, "Encoder.engine")
    with build_encoder_engine(
        args.batch_size, args.sequence_length, config, weights_dict
    ) as serialized_engine:
        print("Saving Engine to {:}".format(encoder_path))
        with open(encoder_path, "wb") as fout:
            fout.write(serialized_engine)
        print("Serializing Encoder Done.")

    decoder_path = os.path.join(args.model_dir, "Decoder.engine")
    

    with build_engine_decoder(
        args.batch_size, args.sequence_length, config, weights_dict
    ) as serialized_engine:
        print("Saving Engine to {:}".format(decoder_path))
        
        with open(decoder_path, "wb") as fout:
            fout.write(serialized_engine)
        print("Serializing Decoder Done.")


if __name__ == "__main__":
    main()
