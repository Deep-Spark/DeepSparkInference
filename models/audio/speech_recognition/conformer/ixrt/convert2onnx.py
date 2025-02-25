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

"""
Build Compute Graph(Fusion Plugin Onnx) From Checkpoints.
"""

import os
import json
import torch
import argparse
import numpy as np
from collections import OrderedDict

from tensorrt.deploy.api import GraphTransform, create_source, create_target
from tensorrt.deploy.ir.data_type import DataType
from tensorrt.deploy.ir.variable import Variable, VariableOptions
from tensorrt.deploy.ir.graph import Graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Compute Graph From Checkpoints.", usage=""
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="conformer",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="checkpont of conformer",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="raw onnx path to save",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="the batch size for test.",
    )
    args = parser.parse_args()
    return args


def add_global_cmvn_op(graph, state_dict, args):
    t = graph

    sub_inputs = [t.make_variable("input", dtype=DataType.FLOAT, shape=(128, 1500, 80))]
    key = "encoder.global_cmvn.mean"
    sub_inputs.append(t.make_variable(name=key, value=state_dict[key]))
    sub_outputs = [t.make_variable("Sub_output_0", dtype=DataType.FLOAT, shape=(128, 1500, 80))]
    t.make_operator(
        "Sub",
        inputs=sub_inputs,
        outputs=sub_outputs,
    )

    mul_inputs = sub_outputs
    key = "encoder.global_cmvn.istd"
    mul_inputs.append(t.make_variable(name=key, value=state_dict[key]))
    mul_outputs = [t.make_variable("Mul_output_0", dtype=DataType.FLOAT, shape=(128, 1500, 80))]
    t.make_operator(
        "Mul",
        inputs=mul_inputs,
        outputs=mul_outputs,
    )

    unsqueeze_inputs = mul_outputs
    unsqueeze_inputs.append(t.make_variable("axes", value=np.array([1], dtype=np.int64)))
    unsqueeze_outputs = [t.make_variable("Unsqueeze_output_0", dtype=DataType.FLOAT, shape=(128, 1, 1500, 80))]
    t.make_operator(
        "Unsqueeze",
        inputs=unsqueeze_inputs,
        outputs=unsqueeze_outputs,
    )


def add_first_submodule_op(graph, state_dict, args):
    """
    The firt submodule part contains follows:
        1.Conv2d+ReLU;
        2.Conv2d+ReLU;
        3.Transpose+Reshape;
        4.MatMul+Add+Mul;
    """

    t = graph
    conv2d0_weight_keys = [
        "encoder.embed.conv.0.weight",
        "encoder.embed.conv.0.bias",
    ]
    conv2d0_attributes = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [3, 3],
        "pads": [0, 0, 0, 0],
        "strides": [2, 2],
    }
    conv2d0_inputs = [t.get_variable("Unsqueeze_output_0")]
    conv2d0_outputs = [t.make_variable("Conv_output_0", dtype=DataType.FLOAT)]

    for key in conv2d0_weight_keys:
        conv2d0_inputs.append(t.make_variable(name=key, value=state_dict[key]))
    t.make_operator(
        "Conv",
        inputs=conv2d0_inputs,
        outputs=conv2d0_outputs,
        **conv2d0_attributes
    )

    relu0_inputs = conv2d0_outputs
    relu0_outputs = [t.make_variable("Relu_output_0", dtype=DataType.FLOAT)]
    t.make_operator(
        "Relu",
        inputs=relu0_inputs,
        outputs=relu0_outputs
    )

    conv2d1_weight_keys = [
        "encoder.embed.conv.2.weight",
        "encoder.embed.conv.2.bias",
    ]
    conv2d1_attributes = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [3, 3],
        "pads": [0, 0, 0, 0],
        "strides": [2, 2],
    }
    conv2d1_inputs = relu0_outputs
    conv2d1_outputs = [t.make_variable("Conv_output_1", dtype=DataType.FLOAT)]

    for key in conv2d1_weight_keys:
        conv2d1_inputs.append(t.make_variable(name=key, value=state_dict[key]))
    t.make_operator(
        "Conv",
        inputs=conv2d1_inputs,
        outputs=conv2d1_outputs,
        **conv2d1_attributes
    )

    relu1_inputs = conv2d1_outputs
    relu1_outputs = [t.make_variable("Relu_output_1", dtype=DataType.FLOAT)]
    t.make_operator(
        "Relu",
        inputs=relu1_inputs,
        outputs=relu1_outputs
    )

    tran_inputs = relu1_outputs
    tran_outputs = [t.make_variable("Transpose_output_0", dtype=DataType.FLOAT)]
    tran_attributes = {"perm": [0, 2, 1, 3]}
    t.make_operator(
        "Transpose",
        inputs=tran_inputs,
        outputs=tran_outputs,
        **tran_attributes
    )

    reshape_inputs = tran_outputs
    reshape_inputs.append(t.make_variable(name="constant_0", value=np.array([args.batch_size, -1, 4864]), dtype=DataType.INT64))
    reshape_outputs = [t.make_variable("Reshape_output_0", dtype=DataType.FLOAT)]
    t.make_operator(
        "Reshape",
        inputs=reshape_inputs,
        outputs=reshape_outputs,
    )

    matmul_inputs = reshape_outputs
    matmul_inputs.append(t.make_variable(name="embed.out.0.weight", value=state_dict["encoder.embed.out.0.weight"].transpose(1, 0)))  # (256,4864)--->(4864,256)
    matmul_outputs = [t.make_variable("MatMul_output_0", dtype=DataType.FLOAT)]
    t.make_operator(
        "MatMul",
        inputs=matmul_inputs,
        outputs=matmul_outputs,
    )

    add_inputs = matmul_outputs
    add_inputs.append(t.make_variable(name="embed.out.0.bias", value=state_dict["encoder.embed.out.0.bias"]))
    add_outputs = [t.make_variable("Add_output_0", dtype=DataType.FLOAT)]
    t.make_operator(
        "Add",
        inputs=add_inputs,
        outputs=add_outputs,
    )

    mul_inputs = add_outputs
    mul_inputs.append(t.make_variable(name="constant_1", value=np.array([16.], dtype=np.float32), dtype=DataType.FLOAT))
    mul_outputs = [t.make_variable("Mul_output_1", dtype=DataType.FLOAT)]
    t.make_operator(
        "Mul",
        inputs=mul_inputs,
        outputs=mul_outputs,
    )


def add_encoder_ff_macaron_op(graph, state_dict, args, index):

    t = graph
    ff_macaron_keys = [
        "encoder.encoders.{}.norm_ff_macaron.weight",
        "encoder.encoders.{}.norm_ff_macaron.bias",
        "encoder.encoders.{}.feed_forward_macaron.w_1.weight",
        "encoder.encoders.{}.feed_forward_macaron.w_1.bias",
        "encoder.encoders.{}.feed_forward_macaron.w_2.weight",
        "encoder.encoders.{}.feed_forward_macaron.w_2.bias",
    ]

    attributes = {
        "in_feature": 256,
        "hidden_size": 2048,
        "act_type": 12,
        "ff_scale": 0.5,
    }

    if index == 0:
        inputs = [graph.get_variable("Mul_output_1")]
    else:
        inputs = [graph.get_variable("norm_final_{}_output".format(index-1))]

    outputs = [t.make_variable("ff_macaron_{}_output".format(index), dtype=DataType.FLOAT)]

    for key in ff_macaron_keys:
        key = key.format(index)
        inputs.append(t.make_variable(name=key, value=state_dict[key].half(), dtype=DataType.FLOAT16))

    t.make_operator(
        "PositionWiseFFNPluginDynamic_IxRT",
        inputs=inputs,
        outputs=outputs,
        **attributes
    )


def add_encoder_mhsa_op(graph, state_dict, args, index):

    t = graph
    mhsa_keys = [
        "encoder.encoders.{}.norm_mha.weight",
        "encoder.encoders.{}.norm_mha.bias",
        "encoder.encoders.{}.self_attn.linear_q.weight",
        "encoder.encoders.{}.self_attn.linear_q.bias",
        "encoder.encoders.{}.self_attn.linear_k.weight",
        "encoder.encoders.{}.self_attn.linear_k.bias",
        "encoder.encoders.{}.self_attn.linear_v.weight",
        "encoder.encoders.{}.self_attn.linear_v.bias",
        "encoder.encoders.{}.self_attn.linear_pos.weight",
        "encoder.encoders.{}.self_attn.pos_bias_u",
        "encoder.encoders.{}.self_attn.pos_bias_v",
        "encoder.encoders.{}.self_attn.linear_out.weight",
        "encoder.encoders.{}.self_attn.linear_out.bias",
    ]

    attributes = {
        "bs": 128,
        "seq_len": 374,
        "n_head": 4,
        "n_feat": 256,
    }

    if index == 0:
        inputs = [
            graph.get_variable("ff_macaron_{}_output".format(index)),
            t.make_variable("mask", dtype=DataType.INT32, shape=(128, 1, 374)),
            t.make_variable("pos_emb", dtype=DataType.FLOAT, shape=(1, 374, 256)),
        ]
    else:
        inputs = [
            graph.get_variable("ff_macaron_{}_output".format(index)),
            graph.get_variable("mask"),
            graph.get_variable("pos_emb"),
        ]

    outputs = [t.make_variable("mhsa_{}_output".format(index), dtype=DataType.FLOAT)]

    for key in mhsa_keys:
        key = key.format(index)
        inputs.append(t.make_variable(name=key, value=state_dict[key].half(), dtype=DataType.FLOAT16))

    t.make_operator(
        "ConformerMultiHeadSelfAttentionPlugin_IxRT",
        inputs=inputs,
        outputs=outputs,
        **attributes
    )


def add_encoder_conv_module_op(graph, state_dict, args, index):

    t = graph
    conv_module_keys = [
        "encoder.encoders.{}.norm_conv.weight",
        "encoder.encoders.{}.norm_conv.bias",
        "encoder.encoders.{}.conv_module.pointwise_conv1.weight",
        "encoder.encoders.{}.conv_module.pointwise_conv1.bias",
        "encoder.encoders.{}.conv_module.depthwise_conv.weight",
        "encoder.encoders.{}.conv_module.depthwise_conv.bias",
        "encoder.encoders.{}.conv_module.norm.weight",
        "encoder.encoders.{}.conv_module.norm.bias",
        "encoder.encoders.{}.conv_module.pointwise_conv2.weight",
        "encoder.encoders.{}.conv_module.pointwise_conv2.bias",
    ]

    attributes = {
        "kernel_size_1": 1,
        "stride_1": 1,
        "odim_1": 512,
        "kernel_size_2": 8,
        "stride_2": 1,
        "odim_2": 256,
        "kernel_size_3": 1,
        "stride_3": 1,
        "odim_3": 256,
    }

    inputs = [
        graph.get_variable("mhsa_{}_output".format(index)),
        graph.get_variable("mask"),
    ]
    outputs = [t.make_variable("conv_module_{}_output".format(index), dtype=DataType.FLOAT)]

    for key in conv_module_keys:
        key = key.format(index)

        if "conv_module.depthwise_conv.weight" in key:
            inputs.append(t.make_variable(name=key, value=state_dict[key].permute(1, 2, 0).half(), dtype=DataType.FLOAT16))
        elif "bias" in key and "norm" not in key:
            inputs.append(t.make_variable(name=key, value=state_dict[key], dtype=DataType.FLOAT))
        else:
            inputs.append(t.make_variable(name=key, value=state_dict[key].half(), dtype=DataType.FLOAT16))

    t.make_operator(
        "ConformerConvModulePlugin_IxRT",
        inputs=inputs,
        outputs=outputs,
        **attributes
    )


def add_encoder_positionwise_ff_op(graph, state_dict, args, index):

    t = graph
    positionwise_ff_keys = [
        "encoder.encoders.{}.norm_ff.weight",
        "encoder.encoders.{}.norm_ff.bias",
        "encoder.encoders.{}.feed_forward.w_1.weight",
        "encoder.encoders.{}.feed_forward.w_1.bias",
        "encoder.encoders.{}.feed_forward.w_2.weight",
        "encoder.encoders.{}.feed_forward.w_2.bias",
    ]

    attributes = {
        "in_feature": 256,
        "hidden_size": 2048,
        "act_type": 12,
        "ff_scale": 0.5,
    }

    inputs = [graph.get_variable('conv_module_{}_output'.format(index))]
    outputs = [t.make_variable("positionwise_ff_{}_output".format(index), dtype=DataType.FLOAT)]

    for key in positionwise_ff_keys:
        key = key.format(index)
        inputs.append(t.make_variable(name=key, value=state_dict[key].half(), dtype=DataType.FLOAT16))

    t.make_operator(
        "PositionWiseFFNPluginDynamic_IxRT",
        inputs=inputs,
        outputs=outputs,
        **attributes
    )


def add_encoder_ln_op(graph, state_dict, args, index):

    t = graph
    ln_keys = [
        "encoder.encoders.{}.norm_final.weight",
        "encoder.encoders.{}.norm_final.bias",
    ]

    attributes = {
        "axis": -1,
        "epsilon": 0.000009999999747378752,
        "stash_type": 1,
    }

    inputs = [graph.get_variable("positionwise_ff_{}_output".format(index))]
    outputs = [t.make_variable("norm_final_{}_output".format(index), dtype=DataType.FLOAT)]

    for key in ln_keys:
        key = key.format(index)
        inputs.append(t.make_variable(name=key, value=state_dict[key].half(), dtype=DataType.FLOAT16))

    t.make_operator(
        "LayerNormalization",
        inputs=inputs,
        outputs=outputs,
        **attributes
    )


def add_final_ln_op(graph, state_dict, args):

    t = graph
    ln_keys = [
        "encoder.after_norm.weight",
        "encoder.after_norm.bias",
    ]

    attributes = {
        "axis": -1,
        "epsilon": 0.000009999999747378752,
        "stash_type": 1,
    }

    inputs = [graph.get_variable("norm_final_11_output")]
    outputs = [t.make_variable("norm_final_output", dtype=DataType.FLOAT)]

    for key in ln_keys:
        inputs.append(t.make_variable(name=key, value=state_dict[key].half(), dtype=DataType.FLOAT16))

    t.make_operator(
        "LayerNormalization",
        inputs=inputs,
        outputs=outputs,
        **attributes
    )


def add_ctc_op(graph, state_dict, args):
    t = graph
    # matmul
    matmul_inputs = [graph.get_variable("norm_final_output")]
    matmul_inputs.append(t.make_variable(name="ctc.ctc_lo.weight", value=state_dict["ctc.ctc_lo.weight"].transpose(1, 0)))   # (4233,256)--->(256,4233)
    matmul_outputs = [t.make_variable("MatMul_output_1", dtype=DataType.FLOAT)]
    t.make_operator(
        "MatMul",
        inputs=matmul_inputs,
        outputs=matmul_outputs,
    )

    add_inputs = matmul_outputs
    add_inputs.append(t.make_variable(name="ctc.ctc_lo.bias", value=state_dict["ctc.ctc_lo.bias"]))
    add_outputs = [t.make_variable("Add_output_1", dtype=DataType.FLOAT)]
    t.make_operator(
        "Add",
        inputs=add_inputs,
        outputs=add_outputs,
    )

    logsoftmax_inputs = add_outputs
    logsoftmax_outputs = [t.make_variable("output", dtype=DataType.FLOAT)]
    attributes = {
        "axis": 2
    }
    t.make_operator(
        "LogSoftmax",
        inputs=logsoftmax_inputs,
        outputs=logsoftmax_outputs,
        **attributes
    )


def main(args):
    graph = Graph()
    transform = GraphTransform(graph)
    state_dict = torch.load(args.model_path)

    # 0. Global CMVN: sub+mul+unsqueeze
    add_global_cmvn_op(transform, state_dict, args)

    # 1. First Submodule: Conv2d+Relu+Transpose+MatMul
    add_first_submodule_op(transform, state_dict, args)

    # 2. Second Submodule: ConformerEncoderLayer: 12 layers
    for i in range(args.num_layers):
        add_encoder_ff_macaron_op(transform, state_dict, args, i)
        add_encoder_mhsa_op(transform, state_dict, args, i)
        add_encoder_conv_module_op(transform, state_dict, args, i)
        add_encoder_positionwise_ff_op(transform, state_dict, args, i)
        add_encoder_ln_op(transform, state_dict, args, i)

    # 3. Third Submodule: FinalNorm
    add_final_ln_op(transform, state_dict, args)

    # 4.Forth Submodule: CTC+LogSoftmax
    add_ctc_op(transform, state_dict, args)

    # 5. set input and output
    graph.add_input(graph.get_variable("input"))
    graph.add_input(graph.get_variable("mask"))
    graph.add_input(graph.get_variable("pos_emb"))
    graph.add_output(graph.get_variable("output"))
    # 5. export onnx file
    create_target(saved_path=args.onnx_path).export(graph)
    print("save onnx: ", args.onnx_path)


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name.lower()
    args.num_layers = 12
    args.hidden_size = 2048
    args.head_num = 4
    args.head_dim = 64
    args.pad_id = 0
    args.inner_size = 3072
    main(args)
