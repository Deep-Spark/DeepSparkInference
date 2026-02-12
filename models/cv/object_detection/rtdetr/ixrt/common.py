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

def save2json(batch_img_id, pred_boxes, json_result,class_trans):
    for i, boxes in enumerate(pred_boxes):
        if boxes is not None:
            image_id = int(batch_img_id[i])
            if image_id == -1:
                continue
            for box in boxes:
                c, p, x, y, w, h = box
                x, y, w, h, p = float(x), float(y), float(w), float(h), float(p)
                c = int(c)
                json_result.append(
                    {
                        "image_id": image_id,
                        "category_id": class_trans[c],
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
