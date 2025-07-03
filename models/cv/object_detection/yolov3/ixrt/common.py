import numpy as np
from tqdm import tqdm

import tensorrt
from cuda import cuda, cudart

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
            for x, y, w, h, c, p in boxes:
                x, y, w, h, p = float(x), float(y), float(w), float(h), float(p)
                c = int(c)
                json_result.append(
                    {
                        "image_id": image_id,
                        "category_id": class_trans[c - 1],
                        "bbox": [x, y, w, h],
                        "score": p,
                    }
                )
def save2json_nonms(batch_img_id, pred_boxes, json_result):
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
    with open(engine_path, "rb") as f:
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
        err, allocation = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
            "nbytes": size,
        }
        print(f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}")
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations