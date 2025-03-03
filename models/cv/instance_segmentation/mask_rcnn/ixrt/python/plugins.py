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
import os

import numpy as np
import tensorrt as trt


def create_roialign_plugin(
    plugin_registry,
    pooler_resolution,
    spatial_scale,
    sampling_ratio,
    num_proposals,
    out_channels,
):
    roialign_plugin_creator = plugin_registry.get_plugin_creator("RoiAlign", "1")
    pooler_resolution_field = trt.PluginField(
        "pooler_resolution",
        np.array([pooler_resolution], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    sampling_ratio_field = trt.PluginField(
        "sampling_ratio",
        np.array([sampling_ratio], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    num_proposals_field = trt.PluginField(
        "num_proposals",
        np.array([num_proposals], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    out_channels_field = trt.PluginField(
        "out_channels",
        np.array([out_channels], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    spatial_scale_field = trt.PluginField(
        "spatial_scale",
        np.array([spatial_scale], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            pooler_resolution_field,
            sampling_ratio_field,
            num_proposals_field,
            out_channels_field,
            spatial_scale_field,
        ]
    )

    roialign_plugin = roialign_plugin_creator.create_plugin(
        "py_roialign_plugin", field_collection
    )
    return roialign_plugin


def create_rpndecode_plugin(
    plugin_registry, top_n, anchors, stride, image_height, image_width
):
    rpndecode_plugin_creator = plugin_registry.get_plugin_creator("RpnDecode", "1")

    top_n_field = trt.PluginField(
        "top_n",
        np.array([top_n], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    stride_field = trt.PluginField(
        "stride",
        np.array([stride], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )
    image_height_field = trt.PluginField(
        "image_height",
        np.array([image_height], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    image_width_field = trt.PluginField(
        "image_width",
        np.array([image_width], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    anchors_field = trt.PluginField(
        "anchors",
        anchors,
        trt.PluginFieldType.FLOAT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            top_n_field,
            stride_field,
            image_height_field,
            image_width_field,
            anchors_field,
        ]
    )

    rpndecode_plugin = rpndecode_plugin_creator.create_plugin(
        "py_rpndecode_plugin", field_collection
    )

    return rpndecode_plugin


def create_rpnnms_plugin(plugin_registry, nms_thresh, post_nms_topk):
    rpnnms_plugin_creator = plugin_registry.get_plugin_creator("RpnNms", "1")

    nms_thresh_field = trt.PluginField(
        "nms_thresh",
        np.array([nms_thresh], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )
    post_nms_topk_field = trt.PluginField(
        "post_nms_topk",
        np.array([post_nms_topk], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            nms_thresh_field,
            post_nms_topk_field,
        ]
    )

    rpnnms_plugin = rpnnms_plugin_creator.create_plugin(
        "py_rpnnms_plugin", field_collection
    )

    return rpnnms_plugin


def create_predictor_decode_plugin(
    plugin_registry, num_boxes, image_height, image_width, bbox_reg_weights
):
    predictor_decode_plugin_creator = plugin_registry.get_plugin_creator(
        "PredictorDecode", "1"
    )

    num_boxes_field = trt.PluginField(
        "num_boxes",
        np.array([num_boxes], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    image_height_field = trt.PluginField(
        "image_height",
        np.array([image_height], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    image_width_field = trt.PluginField(
        "image_width",
        np.array([image_width], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    bbox_reg_weights_field = trt.PluginField(
        "bbox_reg_weights",
        np.array([bbox_reg_weights], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            num_boxes_field,
            image_height_field,
            image_width_field,
            bbox_reg_weights_field,
        ]
    )

    predictor_decode_plugin = predictor_decode_plugin_creator.create_plugin(
        "py_predictor_decode_plugin", field_collection
    )

    return predictor_decode_plugin


def create_batchednms_plugin(
    plugin_registry, nms_method, nms_thresh, detections_per_im
):
    batchednms_plugin_creator = plugin_registry.get_plugin_creator("BatchedNms", "1")

    nms_method_field = trt.PluginField(
        "nms_method",
        np.array([nms_method], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    nms_thresh_field = trt.PluginField(
        "nms_thresh",
        np.array([nms_thresh], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )
    detections_per_im_field = trt.PluginField(
        "detections_per_im",
        np.array([detections_per_im], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            nms_method_field,
            nms_thresh_field,
            detections_per_im_field,
        ]
    )

    batchednms_plugin = batchednms_plugin_creator.create_plugin(
        "py_batchednms_plugin", field_collection
    )

    return batchednms_plugin


def create_maskrcnn_inference_plugin(plugin_registry, detections_per_im, output_size):
    maskrcnn_inference_plugin_creator = plugin_registry.get_plugin_creator(
        "MaskRcnnInference", "1"
    )

    detections_per_im_field = trt.PluginField(
        "detections_per_im",
        np.array([detections_per_im], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    output_size_field = trt.PluginField(
        "output_size",
        np.array([output_size], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )

    field_collection = trt.PluginFieldCollection(
        [
            detections_per_im_field,
            output_size_field,
        ]
    )

    maskrcnn_inference_plugin = maskrcnn_inference_plugin_creator.create_plugin(
        "py_maskrcnn_inference_plugin", field_collection
    )

    return maskrcnn_inference_plugin
