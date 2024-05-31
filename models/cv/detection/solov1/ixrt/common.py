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

import numpy as np
from tqdm import tqdm

import tensorrt
import pycuda.driver as cuda
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

def save2json(batch_img_id, pred_boxes, json_result):
    for i, boxes in enumerate(pred_boxes):
        image_id = int(batch_img_id)
        if boxes is not None:
            x, y, w, h, c, p = boxes
            if image_id!=-1:
                
                x, y, w, h, p = float(x), float(y), float(w), float(h), float(p)
                c = int(c)
                json_result.append(
                    {
                    "image_id": image_id,
                    "category_id": c,
                    "bbox": [x, y, w, h],
                    "score": p,
                    }
                    )

def create_engine_context(engine_path, logger):
    with open(engine_path, "rb") as f, tensorrt.Runtime(logger) as runtime:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context
    return engine, context

def get_io_bindings(engine):
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
        shape = engine.get_binding_shape(i)
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