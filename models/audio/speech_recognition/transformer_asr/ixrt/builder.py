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
import argparse
import torch
from tensorrt.deploy.api import GraphTransform, create_source, create_target
from tensorrt.deploy.ir.data_type import DataType
from tensorrt.deploy.ir.variable import Variable, VariableOptions
from tensorrt.deploy.ir.graph import Graph
from collections import OrderedDict
import math
import re
import glob
import os
from onnx import numpy_helper
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="build ixrt engine", usage=""
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--head_num",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=".tmp.onnx",
        help="",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        required=True,
        help="",
    )
    args = parser.parse_args()
    return args


def add_make_mask_op(graph, state_dict, args):
    attributes = {}

    t = graph
    inputs = [
        graph.make_variable('length_radio', dtype=DataType.FLOAT16),
        graph.make_variable('input', dtype=DataType.FLOAT16),
    ]

    outputs = [t.make_variable("attention_mask", dtype=DataType.INT32)]

    t.make_operator(
        "MakeMaskByRadio_IxRT", inputs=inputs, outputs=outputs, **attributes
    )


def add_custom_linear_op(graph, state_dict, args):
    # Replace CustomFCPluginDynamic_IxRT with MatMul + Add (fused biased GEMM).
    # PyTorch Linear weight is [out, in]; MatMul needs [in, out].
    weight = state_dict["1.custom_src_module.layers.0.w.weight"]
    bias = state_dict["1.custom_src_module.layers.0.w.bias"]
    assert weight.size(0) == bias.size(0)

    t = graph
    weight_var = t.make_variable(
        name="1.custom_src_module.layers.0.w.weight",
        value=weight.half().t().contiguous(),
    )
    bias_var = t.make_variable(
        name="1.custom_src_module.layers.0.w.bias",
        value=bias.half(),
    )

    matmul_out = t.make_variable("custom_src_matmul", dtype=DataType.FLOAT16)
    t.make_operator(
        "MatMul",
        inputs=[graph.get_variable('input'), weight_var],
        outputs=[matmul_out],
    )
    t.make_operator(
        "Add",
        inputs=[matmul_out, bias_var],
        outputs=[t.make_variable("custom_src_output")],
    )


# def add_custom_linear_op(graph, state_dict, args):
#     linear_keys = [
#         "1.custom_src_module.layers.0.w.weight",
#         "1.custom_src_module.layers.0.w.bias"
#     ]
#     attributes = {
#         "linear_dim": state_dict["1.custom_src_module.layers.0.w.weight"].size(0),
#         "hidden_size": state_dict["1.custom_src_module.layers.0.w.weight"].size(1),
#         "has_bias": 1,
#         "act_type": "none",
#     }
#     assert state_dict['1.custom_src_module.layers.0.w.weight'].size(
#         0) == state_dict["1.custom_src_module.layers.0.w.bias"].size(0)
#
#     t = graph
#     inputs = [
#         graph.get_variable('input'),
#     ]
#
#     outputs = [t.make_variable("custom_src_output",dtype=DataType.FLOAT16)]
#     for key in linear_keys:
#         inputs.append(t.make_variable(name=key, value=state_dict[key].half()))
#     t.make_operator(
#         "LinearFP16", inputs=inputs, outputs=outputs, **attributes
#     )


def add_pos_encode_op(graph, state_dict, args):
    """Replace PosEncodeSinCos_IxRT with a precomputed sin/cos PE table + Slice + Add.

    Matches the plugin kernel:
      pe[pos, 2i]   = sin(pos * exp(2i * -log(10000) / d_model))
      pe[pos, 2i+1] = cos(pos * exp(2i * -log(10000) / d_model))
      out = input + pe[:seq_len]
    """
    import numpy as np

    d_model = int(args.hidden_size)
    max_len = int(args.max_seq_len)
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    for pos in range(max_len):
        for j in range(d_model):
            div_term = np.exp((j // 2 * 2) * (-np.log(10000.0)) / d_model)
            pe[pos, j] = np.sin(pos * div_term) if (j % 2 == 0) else np.cos(pos * div_term)
    pe = np.ascontiguousarray(pe.astype(np.float16))

    t = graph
    src = graph.get_variable("custom_src_output")
    pe_table = t.make_variable(
        name="pos_encode_sin_cos_table",
        value=torch.from_numpy(pe),
    )

    src_shape = t.make_variable("custom_src_shape", dtype=DataType.INT64)
    t.make_operator("Shape", inputs=[src], outputs=[src_shape])

    gather_idx = t.make_variable(
        name="pos_encode_shape_idx",
        value=torch.tensor(1, dtype=torch.int64),
    )
    seq_len = t.make_variable("pos_encode_seq_len", dtype=DataType.INT64)
    t.make_operator(
        "Gather",
        inputs=[src_shape, gather_idx],
        outputs=[seq_len],
        axis=0,
    )

    ends = t.make_variable("pos_encode_ends", dtype=DataType.INT64)
    t.make_operator(
        "Unsqueeze",
        inputs=[seq_len],
        outputs=[ends],
        axes=[0],
    )
    starts = t.make_variable(
        name="pos_encode_starts",
        value=torch.tensor([0], dtype=torch.int64),
    )
    axes = t.make_variable(
        name="pos_encode_axes",
        value=torch.tensor([0], dtype=torch.int64),
    )
    steps = t.make_variable(
        name="pos_encode_steps",
        value=torch.tensor([1], dtype=torch.int64),
    )

    pe_sliced = t.make_variable("pos_encode_sliced", dtype=DataType.FLOAT16)
    t.make_operator(
        "Slice",
        inputs=[pe_table, starts, ends, axes, steps],
        outputs=[pe_sliced],
    )

    pe_bsh = t.make_variable("pos_encode_bsh", dtype=DataType.FLOAT16)
    t.make_operator(
        "Unsqueeze",
        inputs=[pe_sliced],
        outputs=[pe_bsh],
        axes=[0],
    )

    t.make_operator(
        "Add",
        inputs=[src, pe_bsh],
        outputs=[t.make_variable("hidden_state", dtype=DataType.FLOAT16)],
    )


def add_transformer_op(graph, state_dict, args):
    enc_tensor_layer_fp16_keys = OrderedDict([
        ["1.encoder.layers.{}.norm1.norm.weight", [args.hidden_size]],
        ["1.encoder.layers.{}.norm1.norm.bias", [args.hidden_size]],
        ["1.encoder.layers.{}.self_att.att.in_proj_weight",
         [args.hidden_size * 3, args.hidden_size]],
        ["1.encoder.layers.{}.self_att.att.in_proj_bias", [args.hidden_size * 3]],
        ["1.encoder.layers.{}.self_att.att.out_proj.weight",
         [args.hidden_size, args.hidden_size]],
        ["1.encoder.layers.{}.self_att.att.out_proj.bias", [args.hidden_size]],
        ["1.encoder.layers.{}.pos_ffn.ffn.0.weight",
         [args.inner_size, args.hidden_size]],
        ["1.encoder.layers.{}.pos_ffn.ffn.0.bias", [args.inner_size]],
        ["1.encoder.layers.{}.pos_ffn.ffn.3.weight",
         [args.hidden_size, args.inner_size]],
        ["1.encoder.layers.{}.pos_ffn.ffn.3.bias", [args.hidden_size]],
        ["1.encoder.layers.{}.norm2.norm.weight", [args.hidden_size]],
        ["1.encoder.layers.{}.norm2.norm.bias", [args.hidden_size]],
    ])
    attributes_legcy = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "head_num": args.head_num,
        "head_dim": args.head_dim,
        "inner_size": args.inner_size,
        "act_type": "gelu",
        "normalize_before": 1,
        "is_fmha": 1,
        "atten_scaler": 1 / math.sqrt(args.head_dim)
    }
    
    
    attributes = {
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "head_num": int(args.head_num),
        "head_dim": int(args.head_dim),
        "inner_size": int(args.inner_size),
        "act_type": 12, #gelu
        "normalize_before": 1,
        "is_fmha": 1,
        "is_casual_mask": 0,
        "concat_after": 0,
        "atten_scaler": 1.0 / math.sqrt(args.head_dim),
        "max_seq_len": int(args.max_seq_len),
        "max_batch_size": int(args.max_batch_size),
        
    }
    
    t = graph
    inputs = [
        graph.get_variable('hidden_state'),
        graph.get_variable('attention_mask'),
    ]
    outputs = [t.make_variable("encoder_out", dtype=DataType.FLOAT16)]
    for layer_id in range(args.num_layers):
        for key, shape in enc_tensor_layer_fp16_keys.items():
            # we need cat qkv gemm's weight and bias
            new_key = key.format(layer_id)
            w = state_dict[new_key]
            if list(w.shape) != shape:
                print("weights shape error!")
                print("key: ", key)
                print("need shape: ", shape)
                print("weight shape: ", w.shape)
                exit(1)
            inputs.append(t.make_variable(name=new_key, value=w.half()))
    t.make_operator(
        "TransformerEncoderFp16_IxRT", inputs=inputs, outputs=outputs, **attributes
    )


def add_layer_norm_op(graph, state_dict, args):
    enc_ln_tensor_fp16_keys = OrderedDict([
        ["1.encoder.norm.norm.weight", [args.hidden_size]],
        ["1.encoder.norm.norm.bias", [args.hidden_size]],
    ])
    attributes = {
        "epsilon": 1e-5,
        "axis": -1,
        "stash_type": 1
    }
    t = graph
    inputs = [
        graph.get_variable('encoder_out'),
    ]
    outputs = [t.make_variable("encoder_ln_out")]
    for key, shape in enc_ln_tensor_fp16_keys.items():
        new_key = key
        w = state_dict[new_key]
        if list(w.shape) != shape:
            print("weights shape error!")
            print("key: ", key)
            print("need shape: ", shape)
            print("weight shape: ", w.shape)
            exit(1)
        inputs.append(t.make_variable(name=new_key, value=w.half()))
    t.make_operator(
        "LayerNormalization", inputs=inputs, outputs=outputs, **attributes
    )


# def add_layer_norm_op(graph, state_dict, args):
#     enc_ln_tensor_fp16_keys = OrderedDict([
#         ["1.encoder.norm.norm.weight", [args.hidden_size]],
#         ["1.encoder.norm.norm.bias", [args.hidden_size]],
#     ])
#     attributes = {
#         "hidden_size": args.hidden_size,
#     }
#     t = graph
#     inputs = [
#         graph.get_variable('encoder_out'),
#     ]
#     outputs = [t.make_variable("encoder_ln_out",dtype=DataType.FLOAT16)]
#     for key, shape in enc_ln_tensor_fp16_keys.items():
#         new_key = key
#         w = state_dict[new_key]
#         if list(w.shape) != shape:
#             print("weights shape error!")
#             print("key: ", key)
#             print("need shape: ", shape)
#             print("weight shape: ", w.shape)
#             exit(1)
#         inputs.append(t.make_variable(name=new_key, value=w.half()))
#     t.make_operator(
#         "LayerNormFp16", inputs=inputs, outputs=outputs, **attributes
#     )

def add_linear_op(graph, state_dict, args):
    # Replaced CustomFCPluginDynamic_IxRT with MatMul + Add (same as add_custom_linear_op).
    # NOTE: This function is currently dead code (commented out in main).
    weight = state_dict["3.w.weight"]
    bias = state_dict["3.w.bias"]
    assert weight.size(0) == bias.size(0)

    t = graph
    weight_var = t.make_variable(
        name="3.w.weight",
        value=weight.half().t().contiguous(),
    )
    bias_var = t.make_variable(
        name="3.w.bias",
        value=bias.half(),
    )

    matmul_out = t.make_variable("lin_matmul", dtype=DataType.FLOAT16)
    t.make_operator(
        "MatMul",
        inputs=[graph.get_variable('encoder_ln_out'), weight_var],
        outputs=[matmul_out],
    )
    t.make_operator(
        "Add",
        inputs=[matmul_out, bias_var],
        outputs=[t.make_variable("lin_output")],
    )


#
# def add_linear_op(graph, state_dict, args):
#     lin_keys = [
#         "3.w.weight",
#         "3.w.bias"
#     ]
#     attributes = {
#         "linear_dim": state_dict["3.w.weight"].size(0),
#         "hidden_size": state_dict["3.w.weight"].size(1),
#         "has_bias": 1,
#         "act_type": "none",
#     }
#     assert state_dict['3.w.weight'].size(0) == state_dict["3.w.bias"].size(0)
#
#     t = graph
#     inputs = [
#         graph.get_variable('encoder_ln_out'),
#     ]
#
#     outputs = [t.make_variable("lin_output",dtype=DataType.FLOAT16)]
#     for key in lin_keys:
#         inputs.append(t.make_variable(name=key, value=state_dict[key].half()))
#     t.make_operator(
#         "LinearFP16", inputs=inputs, outputs=outputs, **attributes
#     )


def add_log_softmax_op(graph, state_dict, args):
    attributes = {
        "axis": "-1",
    }

    t = graph
    inputs = [
        graph.get_variable('lin_output'),
    ]

    outputs = [t.make_variable("log_softmax_output", dtype=DataType.FLOAT16)]

    t.make_operator(
        "LogSoftmax", inputs=inputs, outputs=outputs, **attributes
    )


def add_search_node(graph, state_dict, args):
    attributes = {
        "vocab_size": args.vocab_size,
        "eos_id": args.vocab_size,
        "pad_id": -10000,
        "beam_size": 1,
        "attr1": 1.0,
        "min_decode_ratio": 0.0,
        "max_decode_ratio": 1.0,
        "ctc_weight": 0.40,
        "using_eos_threshold": 0,
        "length_normalization": 1,
    }
    t = graph
    inputs = [
        graph.get_variable('lin_output'),
    ]

    outputs = [t.make_variable("output_tokens", dtype=DataType.INT32)]
    list_value_half = []
    list_key_half = []
    for key in state_dict.keys():
        if "decoder" in key or "custom_tgt_module" in key or "2.w.weight" in key or "2.w.bias" in key:
            list_key_half.append(key)
            list_value_half.append(state_dict[key].half())
    for i, item in enumerate(list_key_half):
        inputs.append(t.make_variable(name=list_key_half[i], value=list_value_half[i]))
    t.make_operator(
        "Search", inputs=inputs, outputs=outputs, **attributes
    )


def get_num_layers(state_dict):
    num_layers = -1
    for key in state_dict:
        layer_id = re.search(
            "1.encoder.layers.([0-9]+).pos_ffn.ffn.0.bias", key)
        if layer_id:
            layer_id = layer_id.group(1)
            num_layers = max(num_layers, int(layer_id) + 1)
    assert num_layers > 0
    return num_layers


def build_engine(onnx_file, engine_file, max_batch_size,max_seq_len):
    # Prefer site-packages ixrt CLI (avoids corex-4.4 ixrt 1.0.x on PATH).
    plugin = os.environ.get(
        "IXRT_PLUGIN_LIB",
        "/usr/local/lib/python3.12/site-packages/ixrt/lib/libixrt_plugin.so",
    )
    cmd = [
        "python3", "-m", "ixrt.cli.ixrtexec",
        "--onnx", onnx_file,
        "--min_shape", "input:1x32x5120,length_radio:1",
        "--opt_shape", "input:8x64x5120,length_radio:8",
        "--max_shape", f"input:{max_batch_size}x{max_seq_len}x5120,length_radio:64",
        "--plugins", plugin,
        "--save_engine", engine_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0 and not os.path.isfile(engine_file):
        raise RuntimeError(f"ixrtexec failed with code {result.returncode}")
    if result.returncode != 0:
        print(f"warning: ixrtexec exited {result.returncode}, but engine exists: {engine_file}")


def main(args):
    graph = Graph()
    transform = GraphTransform(graph)
    ckpt_path = glob.glob(os.path.join(args.ckpt_path, "*/model.ckpt"))[0]
    print("load ckpt from: ", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # print([i for i in state_dict ])
    # print(state_dict['3.w.bias'])
    args.hidden_size = state_dict['1.encoder.layers.0.norm1.norm.weight'].size(
        0)
    args.head_dim = args.hidden_size / args.head_num
    args.inner_size = state_dict['1.encoder.layers.0.pos_ffn.ffn.0.bias'].size(
        0)
    args.vocab_size = state_dict['3.w.weight'].size(0)

    args.num_layers = get_num_layers(state_dict)

    args.src_len = state_dict["1.custom_src_module.layers.0.w.weight"].size(1)

    # args.num_layers = 1
    add_make_mask_op(transform, state_dict, args)
    add_custom_linear_op(transform, state_dict, args)
    add_pos_encode_op(transform, state_dict, args)
    add_transformer_op(transform, state_dict, args)
    add_layer_norm_op(transform, state_dict, args)
    # add_linear_op(transform, state_dict, args)
    # add_log_softmax_op(transform, state_dict, args)
    # add_search_node(transform, state_dict, args)

    # IO attributes
    length_radio = graph.get_variable('length_radio')
    length_radio.set_shape(["batch_size"])
    length_radio.dtype = "float16"
    graph.add_input(length_radio)

    input = graph.get_variable('input')
    input.set_shape(["batch_size", "seq_len", "src_len"])
    input.dtype = "float16"
    graph.add_input(input)

    output = graph.get_variable('encoder_ln_out')
    output.dtype = "float16"
    graph.add_output(output)

    create_target(saved_path=args.onnx_path).export(graph)

    build_engine(args.onnx_path, args.engine_path, args.max_batch_size, args.max_seq_len)
    print("save engine: ", args.engine_path)


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    main(args)

"""
python3 builder.py \
--ckpt_path results/transformer/8886/save \
--head_num 4 \
--max_batch_size 64  \
--max_seq_len 1024 \
--engine_path transformer.engine
"""
