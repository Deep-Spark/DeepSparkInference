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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import struct

import numpy as np
import tensorrt

num_blocks_per_stage_r50 = [3, 4, 6, 3]


def basic_stem(network, weight_map, lname, input, out_channels, group_num=1):
    conv1 = network.add_convolution_nd(
        input,
        out_channels,
        kernel_shape=(7, 7),
        kernel=weight_map[lname + ".conv1.weight"],
        bias=weight_map[lname + ".conv1.bias"],
    )
    assert conv1
    conv1.stride_nd = (2, 2)
    conv1.padding_nd = (3, 3)
    conv1.num_groups = group_num

    r1 = network.add_activation(conv1.get_output(0), type=tensorrt.ActivationType.RELU)
    assert r1

    max_pool2d = network.add_pooling_nd(
        r1.get_output(0), tensorrt.PoolingType.MAX, (3, 3)
    )
    max_pool2d.stride_nd = (2, 2)
    max_pool2d.padding_nd = (1, 1)

    return max_pool2d


def basic_block(network, weight_map, lname, input, in_channels, out_channels, stride=1):

    conv1 = network.add_convolution_nd(
        input,
        out_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[lname + ".conv1.weight"],
        bias=weight_map[lname + ".conv1.bias"],
    )
    assert conv1
    conv1.stride_nd = (stride, stride)
    conv1.padding_nd = (1, 1)

    r1 = network.add_activation(conv1.get_output(0), type=tensorrt.ActivationType.RELU)
    assert r1

    conv2 = network.add_convolution_nd(
        r1.get_output(0),
        out_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[lname + ".conv2.weight"],
        bias=weight_map[lname + ".conv2.bias"],
    )
    assert conv2
    conv2.stride_nd = (1, 1)
    conv2.padding_nd = (1, 1)

    # shortcut
    shortcut_out = None
    if in_channels != out_channels:
        shortcut = network.add_convolution_nd(
            input,
            out_channels,
            kernel_shape=(1, 1),
            kernel=weight_map[lname + ".shortcut.weight"],
            bias=weight_map[lname + ".shortcut.bias"],
        )
        assert shortcut
        shortcut.stride_nd = (stride, stride)
        shortcut_out = shortcut.get_output(0)
    else:
        shortcut_out = input

    # add
    ew = network.add_elementwise(
        conv2.get_output(0), shortcut_out, tensorrt.ElementWiseOperation.SUM
    )
    assert ew

    r3 = network.add_activation(ew.get_output(0), type=tensorrt.ActivationType.RELU)
    assert r3

    return r3.get_output(0)


def bottleneck_block(
    network,
    weight_map,
    lname,
    input,
    in_channels,
    bottleneck_channels,
    out_channels,
    stride=1,
    dilation=1,
    group_num=1,
):
    stride_1x1 = stride
    stride_3x3 = 1
    # conv1
    conv1 = network.add_convolution_nd(
        input,
        bottleneck_channels,
        kernel_shape=(1, 1),
        kernel=weight_map[lname + ".conv1.weight"],
        bias=weight_map[lname + ".conv1.bias"],
    )
    assert conv1
    conv1.stride_nd = (stride_1x1, stride_1x1)
    conv1.num_groups = group_num

    r1 = network.add_activation(conv1.get_output(0), type=tensorrt.ActivationType.RELU)

    # conv2
    conv2 = network.add_convolution_nd(
        r1.get_output(0),
        bottleneck_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[lname + ".conv2.weight"],
        bias=weight_map[lname + ".conv2.bias"],
    )
    assert conv2
    conv2.stride_nd = (stride_3x3, stride_3x3)
    conv2.padding_nd = (dilation, dilation)
    conv2.dilation_nd = (dilation, dilation)
    conv2.num_groups = group_num

    r2 = network.add_activation(conv2.get_output(0), type=tensorrt.ActivationType.RELU)

    # conv3
    conv3 = network.add_convolution_nd(
        r2.get_output(0),
        out_channels,
        kernel_shape=(1, 1),
        kernel=weight_map[lname + ".conv3.weight"],
        bias=weight_map[lname + ".conv3.bias"],
    )
    assert conv3
    conv3.stride_nd = (stride_3x3, stride_3x3)
    conv3.num_groups = group_num

    # shortcut
    shortcut_out = None
    if in_channels != out_channels:
        shortcut = network.add_convolution_nd(
            input,
            out_channels,
            kernel_shape=(1, 1),
            kernel=weight_map[lname + ".shortcut.weight"],
            bias=weight_map[lname + ".shortcut.bias"],
        )
        assert shortcut
        shortcut.stride_nd = (stride, stride)
        shortcut.num_groups = group_num
        shortcut_out = shortcut.get_output(0)
    else:
        shortcut_out = input

    # add
    ew = network.add_elementwise(
        conv3.get_output(0), shortcut_out, tensorrt.ElementWiseOperation.SUM
    )
    assert ew

    r3 = network.add_activation(ew.get_output(0), type=tensorrt.ActivationType.RELU)
    assert r3

    return r3.get_output(0)


def make_stage(
    network,
    weight_map,
    lname,
    input,
    stage,
    in_channels,
    bottleneck_channels,
    out_channels,
    first_stride=1,
    dilation=1,
):
    out = input
    for i in range(stage):
        layerName = lname + "." + str(i)
        stride = first_stride if i == 0 else 1
        out = bottleneck_block(
            network,
            weight_map,
            layerName,
            out,
            in_channels,
            bottleneck_channels,
            out_channels,
            stride,
            dilation,
        )
        in_channels = out_channels

    return out


def build_resnet50(
    network,
    weight_map,
    input,
    stem_out_channels,
    bottleneck_channels,
    res2_out_channels,
    res5_dilation=1,
):
    out_channels = res2_out_channels
    out = None

    stem = basic_stem(network, weight_map, "backbone.stem", input, stem_out_channels)
    out = stem.get_output(0)

    # res
    for i in range(3):
        dilation = res5_dilation if i == 3 else 1
        # first_stride = 1 if (i == 0) and (dilation == 2) else 2
        first_stride = 1 if (i == 0 or (i == 3 and dilation == 2)) else 2
        out = make_stage(
            network,
            weight_map,
            "backbone.res" + str(i + 2),
            out,
            num_blocks_per_stage_r50[i],
            stem_out_channels,
            bottleneck_channels,
            out_channels,
            first_stride,
            dilation,
        )
        stem_out_channels = out_channels
        bottleneck_channels *= 2
        out_channels *= 2

    return out
