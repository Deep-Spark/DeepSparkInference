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
import os
import cv2
import glob
import time
import numpy as np
from tqdm import tqdm

import tensorrt
import pycuda.driver as cuda


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

# input  : [bsz, box_num, 5(cx, cy, w, h, conf) + class_num(prob[0], prob[1], ...)]
# output : [bsz, box_num, 6(left_top_x, left_top_y, right_bottom_x, right_bottom_y, class_id, max_prob*conf)]
def box_class85to6(input):
    center_x_y = input[:, :2]
    side = input[:, 2:4]
    conf = input[:, 4:5]
    class_id = np.argmax(input[:, 5:], axis = -1)
    class_id = class_id.astype(np.float32).reshape(-1, 1) + 1
    max_prob = np.max(input[:, 5:], axis = -1).reshape(-1, 1)
    x1_y1 = center_x_y - 0.5 * side
    x2_y2 = center_x_y + 0.5 * side
    nms_input = np.concatenate([x1_y1, x2_y2, class_id, max_prob*conf], axis = -1)
    return nms_input

def save2json(batch_img_id, pred_boxes, json_result, class_trans):
    for i, boxes in enumerate(pred_boxes):
        if boxes is not None:
            image_id = int(batch_img_id[i])
            # have no target
            if image_id == -1:
                continue

            for x1, y1, x2, y2, _, p, c in boxes:
                x1, y1, x2, y2, p = float(x1), float(y1), float(x2), float(y2), float(p)
                c = int(c)
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                json_result.append(
                    {
                        "image_id": image_id,
                        "category_id": class_trans[c - 1],
                        "bbox": [x, y, w, h],
                        "score": p,
                    }
                )

################## About TensorRT #################
def create_engine_context(engine_path, logger):
    with open(engine_path, "rb") as f:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

    return engine, context

def setup_io_bindings(engine, context):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []

    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = np.dtype(tensorrt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        allocation = cuda.mem_alloc(size)
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
        }
        # print(f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}")
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations
##########################################################


################## About Loading Dataset #################
def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

def prepare_batch(images_path, bs=16, input_size=(608, 608)):

    width, height = input_size

    batch_names = []
    batch_images = []
    batch_shapes = []

    temp_names = []
    temp_images = []
    temp_shapes = []

    for i, image_path in tqdm(enumerate(images_path), desc="Loading coco data"):
        name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.
        custom_image = np.expand_dims(custom_image, axis=0)

        if i != 0 and i % bs == 0:
            batch_names.append(temp_names)
            batch_images.append(np.concatenate(temp_images, axis=0))
            batch_shapes.append(temp_shapes)

            temp_names = [name]
            temp_images = [custom_image]
            temp_shapes = [(h, w)]
        else:
            temp_names.append(name)
            temp_images.append(custom_image)
            temp_shapes.append((h, w))

    return batch_names, batch_images, batch_shapes
##########################################################


################## About Operating box #################
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def scale_boxes(net_shape, boxes, ori_shape, use_letterbox=False):
    # Rescale boxes (xyxy) from net_shape to ori_shape

    if use_letterbox:

        gain = min(
            net_shape[0] / ori_shape[0], net_shape[1] / ori_shape[1]
        )  # gain  = new / old
        pad = (net_shape[1] - ori_shape[1] * gain) / 2, (
            net_shape[0] - ori_shape[0] * gain
        ) / 2.0

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
    else:
        x_scale, y_scale = net_shape[1] / ori_shape[1], net_shape[0] / ori_shape[0]

        boxes[:, 0] /= x_scale
        boxes[:, 1] /= y_scale
        boxes[:, 2] /= x_scale
        boxes[:, 3] /= y_scale

    clip_boxes(boxes, ori_shape)
    return boxes

def clip_boxes(boxes, shape):

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
##########################################################


################## About pre and post processing #########
def pre_processing(src_img, imgsz=608):
    resized = cv2.resize(src_img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    in_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    in_img = np.transpose(in_img, (2, 0, 1)).astype(np.float32)
    in_img = np.expand_dims(in_img, axis=0)
    in_img /= 255.0
    return in_img

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(img, conf_thresh, nms_thresh, output, num_classes=80):

    # [batch, num, 1, 4]
    box_array = output[:, :, :4]
    # [batch, num, 2]
    class_confs = output[:, :, 4:]

    max_conf = class_confs[:, :, 1]
    max_id = class_confs[:, :, 0]

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                  ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    return bboxes_batch
##########################################################

