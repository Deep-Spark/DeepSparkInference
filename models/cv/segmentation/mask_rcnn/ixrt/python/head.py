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

import ctypes
import math
import os

import numpy as np
import tensorrt as trt
from backbone import make_stage
from tensorrt import Dims, DimsHW

from plugins import *

PLUGIN_LIB = "../plugins/build/libmaskrcnn.so"
if os.path.exists(PLUGIN_LIB):
    ctypes.CDLL(PLUGIN_LIB)

plugin_registry = trt.get_plugin_registry()

ANCHOR_SIZES = [32, 64, 128, 256, 512]
ASPECT_RATIOS = [0.5, 1.0, 2.0]
BBOX_REG_WEIGHTS = [10.0, 10.0, 5.0, 5.0]
BATCHSIZE = 1
PRE_NMS_TOP_K_TEST = 6000
NMS_THRESH_TEST = 0.5
POST_NMS_TOPK = 1000
DETECTIONS_PER_IMAGE = 100
POOLER_RESOLUTION = 14
STRIDES = 16
FEATURE_CHANNEL = 1024
SAMPLING_RATIO = 0
FEATURE_H = 50
FEATURE_W = 67
RPN_NMS_THRESH = 0.5
INPUT_H = 800
INPUT_W = 1067
RES2_OUT_CHANNELS = 256
NUM_CLASSES = 80
NMS_METHOD = 1
MASK_ON = True


def generate_anchors(anchor_sizes, aspect_ratios):
    res = []
    for anchor_size in anchor_sizes:
        area = anchor_size * anchor_size
        for ar in aspect_ratios:
            w = math.sqrt(area / ar)
            h = ar * w
            res.extend([-w / 2, -h / 2, w / 2, h / 2])
    return np.ascontiguousarray(np.array(res, dtype=np.float32))


def add_rpn(network, weight_map, features):
    num_anchors = len(ANCHOR_SIZES) * len(ASPECT_RATIOS)
    box_dim = 4

    rpn_in_channel = features.shape[1]
    # rpn head conv
    rpn_head_conv = network.add_convolution_nd(
        features,
        rpn_in_channel,
        DimsHW(3, 3),
        weight_map["proposal_generator.rpn_head.conv.weight"],
        weight_map["proposal_generator.rpn_head.conv.bias"],
    )
    rpn_head_conv.stride_nd = DimsHW(1, 1)
    rpn_head_conv.padding_nd = DimsHW(1, 1)

    rpn_head_relu = network.add_activation(
        rpn_head_conv.get_output(0), trt.ActivationType.RELU
    )
    # objectness logits
    rpn_head_logits = network.add_convolution_nd(
        rpn_head_relu.get_output(0),
        num_anchors,
        DimsHW(1, 1),
        weight_map["proposal_generator.rpn_head.objectness_logits.weight"],
        weight_map["proposal_generator.rpn_head.objectness_logits.bias"],
    )
    rpn_head_logits.stride_nd = DimsHW(1, 1)
    # anchor deltas
    rpn_head_deltas = network.add_convolution_nd(
        rpn_head_relu.get_output(0),
        num_anchors * box_dim,
        DimsHW(1, 1),
        weight_map["proposal_generator.rpn_head.anchor_deltas.weight"],
        weight_map["proposal_generator.rpn_head.anchor_deltas.bias"],
    )
    rpn_head_deltas_dim = rpn_head_deltas.get_output(0).shape

    anchors = generate_anchors(ANCHOR_SIZES, ASPECT_RATIOS)
    rpn_decode_plugin = create_rpndecode_plugin(
        plugin_registry, PRE_NMS_TOP_K_TEST, anchors, STRIDES, INPUT_H, INPUT_W
    )
    faster_decode_inputs = [
        rpn_head_logits.get_output(0),
        rpn_head_deltas.get_output(0),
    ]

    rpn_decode_layer = network.add_plugin_v2(faster_decode_inputs, rpn_decode_plugin)

    nms_input = [rpn_decode_layer.get_output(0), rpn_decode_layer.get_output(1)]
    nms_plugin = create_rpnnms_plugin(plugin_registry, RPN_NMS_THRESH, POST_NMS_TOPK)
    nms_layer = network.add_plugin_v2(nms_input, nms_plugin)

    return nms_layer.get_output(0)


def shared_roi_transform(network, weight_map, proposals, features, num_proposals):
    roi_inputs = [proposals, features]
    output_channels = features.shape[1]
    roi_align_plugin = create_roialign_plugin(
        plugin_registry,
        POOLER_RESOLUTION,
        1 / STRIDES,
        SAMPLING_RATIO,
        num_proposals,
        output_channels,
    )
    roi_align_layer = network.add_plugin_v2(roi_inputs, roi_align_plugin)

    reshape = network.add_shuffle(roi_align_layer.get_output(0))
    dims = roi_align_layer.get_output(0).shape
    reshape.reshape_dims = Dims([dims[0] * dims[1], dims[2], dims[3], dims[4]])

    stage_in_channels = reshape.get_output(0).shape
    box_features = make_stage(
        network,
        weight_map,
        "roi_heads.res5",
        reshape.get_output(0),
        3,
        stage_in_channels[1],
        512,
        RES2_OUT_CHANNELS * 8,
        2,
    )
    return box_features


def add_box_head(network, weight_map, proposals, features, instances):
    box_features = shared_roi_transform(
        network, weight_map, proposals, features, POST_NMS_TOPK
    )
    box_features_mean = network.add_reduce(
        box_features, trt.ReduceOperation.AVG, 12, True
    )

    scores = network.add_fully_connected(
        box_features_mean.get_output(0),
        NUM_CLASSES + 1,
        weight_map["roi_heads.box_predictor.cls_score.weight"],
        weight_map["roi_heads.box_predictor.cls_score.bias"],
    )
    probs = network.add_softmax(scores.get_output(0))
    probs_dim = probs.get_output(0).shape
    score_slice = network.add_slice(
        probs.get_output(0),
        Dims([0, 0, 0, 0]),
        Dims([probs_dim[0], probs_dim[1] - 1, 1, 1]),
        Dims([1, 1, 1, 1]),
    )

    proposal_deltas = network.add_fully_connected(
        box_features_mean.get_output(0),
        NUM_CLASSES * 4,
        weight_map["roi_heads.box_predictor.bbox_pred.weight"],
        weight_map["roi_heads.box_predictor.bbox_pred.bias"],
    )
    score_slice_reshape = network.add_shuffle(score_slice.get_output(0))

    dims0 = score_slice.get_output(0).shape
    score_slice_reshape.reshape_dims = Dims(
        [BATCHSIZE, dims0[0] // BATCHSIZE, dims0[1], 1, 1]
    )

    proposal_deltas_reshape = network.add_shuffle(proposal_deltas.get_output(0))
    dims1 = proposal_deltas.get_output(0).shape
    proposal_deltas_reshape.reshape_dims = Dims(
        [BATCHSIZE, dims1[0] // BATCHSIZE, dims1[1], 1, 1]
    )

    predictor_decode_input = [
        score_slice_reshape.get_output(0),
        proposal_deltas_reshape.get_output(0),
        proposals,
    ]
    predictor_decode_plugin = create_predictor_decode_plugin(
        plugin_registry, probs_dim[0], INPUT_H, INPUT_W, BBOX_REG_WEIGHTS
    )
    predictor_decode_layer = network.add_plugin_v2(
        predictor_decode_input, predictor_decode_plugin
    )

    nms_input = [
        predictor_decode_layer.get_output(0),
        predictor_decode_layer.get_output(1),
        predictor_decode_layer.get_output(2),
    ]
    batched_nms_plugin = create_batchednms_plugin(
        plugin_registry, NMS_METHOD, NMS_THRESH_TEST, DETECTIONS_PER_IMAGE
    )
    batched_nms_layer = network.add_plugin_v2(nms_input, batched_nms_plugin)

    instances.extend(
        [
            batched_nms_layer.get_output(0),
            batched_nms_layer.get_output(1),
            batched_nms_layer.get_output(2),
        ]
    )


def add_mask_head(network, weight_map, features, instances, out_channels=256):
    mask_features = shared_roi_transform(
        network, weight_map, instances[1], features, DETECTIONS_PER_IMAGE
    )

    mask_deconv = network.add_deconvolution_nd(
        mask_features,
        out_channels,
        DimsHW(2, 2),
        weight_map["roi_heads.mask_head.deconv.weight"],
        weight_map["roi_heads.mask_head.deconv.bias"],
    )
    mask_deconv.stride_nd = DimsHW(2, 2)
    deconv_relu = network.add_activation(
        mask_deconv.get_output(0), trt.ActivationType.RELU
    )

    predictor = network.add_convolution_nd(
        deconv_relu.get_output(0),
        NUM_CLASSES,
        DimsHW(1, 1),
        weight_map["roi_heads.mask_head.predictor.weight"],
        weight_map["roi_heads.mask_head.predictor.bias"],
    )
    predictor.stride_nd = DimsHW(1, 1)

    predictor_reshape = network.add_shuffle(predictor.get_output(0))
    dims1 = predictor.get_output(0).shape
    predictor_reshape.reshape_dims = Dims(
        [BATCHSIZE, dims1[0] // BATCHSIZE, dims1[1], dims1[2], dims1[3]]
    )

    if NUM_CLASSES == 1:
        mask_probs_pred = network.add_activation(
            predictor_reshape.get_output(0), trt.ActivationType.SIGMOID
        )
        instances.append(mask_probs_pred.get_output(0))
    else:
        mask_rcnn_inference_inputs = [instances[2], predictor_reshape.get_output(0)]
        mask_rcnn_inference_plugin = create_maskrcnn_inference_plugin(
            plugin_registry, DETECTIONS_PER_IMAGE, POOLER_RESOLUTION
        )
        mask_rcnn_inference_layer = network.add_plugin_v2(
            mask_rcnn_inference_inputs, mask_rcnn_inference_plugin
        )
        instances.append(mask_rcnn_inference_layer.get_output(0))


def add_roi_heads(network, weight_map, proposals, features):
    instances = []
    add_box_head(network, weight_map, proposals, features, instances)
    if MASK_ON:
        add_mask_head(network, weight_map, features, instances)

    return instances
