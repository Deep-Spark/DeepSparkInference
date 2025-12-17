import cv2
import numpy as np

import torch
import torchvision

coco80_to_coco91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90
]

coco80_to_coco91_dict = {idx: i for idx, i in enumerate(coco80_to_coco91)}
coco91_to_coco80_dict = {i: idx for idx, i in enumerate(coco80_to_coco91)}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    # import pdb;pdb.set_trace()
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, ratio, (dw, dh)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(),
                     y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([
            np.interp(x, xp, s[:, i]) for i in range(2)
        ]).reshape(2, -1).T  # segment xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    return segments


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(boxes, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x
        boxes[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        boxes[:, 0] = boxes[:, 0].clip(0, shape[1])  # x
        boxes[:, 1] = boxes[:, 1].clip(0, shape[0])  # y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(
            prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    # device = prediction.device
    # mps = 'mps' in device.type  # Apple MPS
    # if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
    #     prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask),
                          1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(
                descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:,
                                        4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n <
                      3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        # if mps:
        #     output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #     break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
                '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
                '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
                'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


def process_batch(detections, labels, iouv):
    """
    example:

    correct = process_batch(predn, labelsn, iouv)

    predn: torch.Size([37, 6])
    tensor([[0.00000e+00, 2.54544e+02, 1.23906e+02, 5.63799e+02, 9.03213e-01, 0.00000e+00],
        [5.50954e+01, 3.97058e+02, 1.00450e+02, 4.44562e+02, 7.95156e-01, 3.80000e+01],
        [5.26250e+01, 3.97192e+02, 1.00602e+02, 4.42674e+02, 3.27940e-01, 0.00000e+00],
        [1.14922e+00, 2.53101e+02, 1.19968e+02, 5.59550e+02, 3.18507e-01, 3.80000e+01],
        [4.14812e+01, 3.35122e+02, 1.03381e+02, 5.03794e+02, 1.44468e-01, 0.00000e+00],
        [4.14812e+01, 3.35122e+02, 1.03381e+02, 5.03794e+02, 5.82885e-02, 3.80000e+01],
        [5.34525e+00, 3.20239e+02, 1.08178e+02, 5.05265e+02, 2.66997e-02, 0.00000e+00],
        [4.07626e+00, 0.00000e+00, 4.61923e+01, 9.27553e+00, 1.85876e-02, 0.00000e+00],
        [0.00000e+00, 0.00000e+00, 3.20227e+01, 9.35835e+00, 1.24020e-02, 0.00000e+00],
        [5.34525e+00, 3.20239e+02, 1.08178e+02, 5.05265e+02, 1.20568e-02, 3.80000e+01],
        [1.73669e+01, 0.00000e+00, 4.78601e+01, 8.00911e+00, 1.17472e-02, 0.00000e+00],
        [0.00000e+00, 4.21593e-01, 2.05969e+01, 8.87799e+00, 1.01791e-02, 0.00000e+00],
        [4.77588e+01, 3.79223e+02, 9.97695e+01, 4.40225e+02, 6.46325e-03, 3.80000e+01],
        [6.02026e+00, 3.89793e+02, 1.06110e+02, 4.33794e+02, 6.31573e-03, 3.80000e+01],
        [6.74133e-01, 0.00000e+00, 4.91198e+01, 8.86116e+00, 5.98140e-03, 0.00000e+00],
        [2.12402e-01, 2.51346e+02, 7.27914e+01, 5.56545e+02, 4.98825e-03, 0.00000e+00],
        [0.00000e+00, 2.13631e-01, 1.25884e+01, 8.68380e+00, 4.47612e-03, 0.00000e+00],
        [6.35330e+01, 3.98378e+02, 9.95361e+01, 4.44679e+02, 4.13858e-03, 3.20000e+01],
        [2.36788e+01, 0.00000e+00, 5.18657e+01, 6.79403e+00, 3.42408e-03, 0.00000e+00],
        [2.11297e+00, 0.00000e+00, 3.05802e+01, 7.74192e+00, 3.11905e-03, 0.00000e+00],
        [2.00028e+01, 2.34144e+02, 1.75109e+02, 5.84247e+02, 2.58519e-03, 0.00000e+00],
        [0.00000e+00, 0.00000e+00, 1.66594e+01, 7.45438e+00, 2.43317e-03, 0.00000e+00],
        [4.78100e+01, 3.83032e+02, 7.78318e+01, 4.11393e+02, 1.93834e-03, 3.80000e+01],
        [9.23322e+00, 0.00000e+00, 3.75083e+01, 7.47940e+00, 1.84539e-03, 0.00000e+00],
        [4.38248e+01, 3.81181e+02, 8.79560e+01, 4.25744e+02, 1.73720e-03, 3.80000e+01],
        [5.60219e+01, 3.95002e+02, 1.00572e+02, 4.43532e+02, 1.58277e-03, 3.50000e+01],
        [5.58219e+01, 3.91919e+02, 1.00614e+02, 4.45245e+02, 1.46675e-03, 5.60000e+01],
        [2.36203e+01, 0.00000e+00, 6.64134e+01, 8.02699e+00, 1.45388e-03, 0.00000e+00],
        [5.07014e+01, 3.86152e+02, 6.96711e+01, 4.06287e+02, 1.41022e-03, 3.80000e+01],
        [5.60219e+01, 3.95002e+02, 1.00572e+02, 4.43532e+02, 1.37910e-03, 2.90000e+01],
        [7.68276e+01, 3.98319e+02, 1.01517e+02, 4.43442e+02, 1.32950e-03, 3.80000e+01],
        [1.14922e+00, 2.53101e+02, 1.19968e+02, 5.59550e+02, 1.30156e-03, 3.50000e+01],
        [1.14922e+00, 2.53101e+02, 1.19968e+02, 5.59550e+02, 1.28391e-03, 3.40000e+01],
        [1.97944e+02, 6.47475e+01, 2.37046e+02, 1.36132e+02, 1.23386e-03, 3.80000e+01],
        [1.31996e+01, 0.00000e+00, 5.24565e+01, 1.01653e+01, 1.22542e-03, 0.00000e+00],
        [4.02082e+01, 3.59193e+02, 1.31879e+02, 4.77750e+02, 1.16416e-03, 3.80000e+01],
        [3.01674e+01, 0.00000e+00, 5.76949e+01, 7.27436e+00, 1.08308e-03, 0.00000e+00]])

    labelsn: torch.Size([2, 5])
    tensor([[ 38.00000,  60.12997, 399.08969, 102.09995, 443.49973],
        [  0.00000,   0.00000, 257.44031, 122.95984, 559.06012]])

    iouv: torch.Size([10])
    tensor([0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000])

    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class, N表示当前图片上检测出的box数
        labels (array[M, 5]), class, x1, y1, x2, y2, M表示当前图片上的gt的box数
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    ## correct的每一行表示，当前预测的box在不同的iou阈值上，被认为是TP或FP
    correct = np.zeros(
        (detections.shape[0], iouv.shape[0])).astype(bool)  # N * 10

    # 对于每条label，判断其真实box和预测的box的iou
    iou = box_iou(labels[:, 1:], detections[:, :4])  # M * N

    # 对于每条label，判断其真实class和预测的class是否相当
    correct_class = labels[:, 0:1] == detections[:, 5]  # M * N

    # 对于每个iou阈值进行循环，计算
    for i in range(len(iouv)):
        ## 等价于torch.nonzero((iou >= iouv[i]) & correct_class, as_tuple=True)
        # x[0], x[1]代表(iou >= iouv[i]) & correct_class的非0元素的横坐标和纵坐标的列表
        #
        x = torch.where((iou >= iouv[i])
                        & correct_class)  # IoU > threshold and classes match

        if x[0].shape[0]:
            ## matchs shape (len(x[0]), 3)， 每一行元素，代表在iou中的(横坐标, 纵坐标, iou)
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:

                ## 按iou从高到低排列
                matches = matches[matches[:, 2].argsort()[::-1]]

                ## 去除纵坐标上的重复元素
                matches = matches[np.unique(matches[:, 1],
                                            return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]

                ## 去除横坐标上的重复元素
                matches = matches[np.unique(matches[:, 0],
                                            return_index=True)[1]]

            ## 横坐标没有意义，仅仅表示数量，纵坐标对应的是在N维度的坐标
            # 此处更新的是correct的第i列，即对于第i个iou阈值的，TP的标为true
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    # 由于折现积分比较困难，需要补全，参考https://zhuanlan.zhihu.com/p/70667071
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        ## 求得积分结果
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    else:  # 'continuous'
        i = np.where(
            mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(
        len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp,
                 conf,
                 pred_cls,
                 target_cls,
                 plot=False,
                 save_dir='.',
                 names=(),
                 eps=1e-16,
                 prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    ## 按置信度从大到小排列，选择置信度最高的index
    i = np.argsort(-conf)
    ## 按置信度从大到小的顺序，重排列 tp, conf, pred_cls
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    ## 找到不同的class的数目，即需要计算ap的class数，coco的话必定在80以下
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting

    ## 初始化ap, precision, recall
    ## 这里用线性插值来构造pr曲线
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros(
        (nc, 1000))
    # 对每个类别计算
    for ci, c in enumerate(unique_classes):

        # 找到预测class和该class一致的坐标
        # 用于后面获取tp[i]，表示符合该class的不同置信度的TP结果
        i = pred_cls == c

        n_l = nt[ci]  # number of labels  ## 该class的label数目
        n_p = i.sum()  # number of predictions ## 预测为该class的pred数目
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        ## 计算当前class在不同置信度阈值下的的TP和FP的数目
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        ## n_l表示当前class正样本的总数
        recall = tpc / (n_l + eps)  # recall curve
        ## 线性插值置信度阈值为50时的该类的recall曲线
        r[ci] = np.interp(-px, -conf[i], recall[:, 0],
                          left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        ## 线性插值置信度阈值为50时的该类的precision曲线
        p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                          left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            # 当前class的第j个置信度阈值情况下的ap
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            # if plot and j == 0:
            #     py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    ## p和r都是线性插值0到1之间1000个点对应的值，0~1表示iou阈值
    ## f1用来计算调和平均
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items()
             if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    # if plot:
    #     plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
    #     plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
    #     plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
    #     plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    ## 由于随着iou阈值提高，p单调上升，r单调下降，而f1值先升后降，选择f1最大时候的
    ## f1的shape时 class * 100，f1.mean(0)的shape时1000，相当于在每个iou阈值上，取所有class的f1的平均
    ## i则是用来取得平滑后的1000个置信度中使得平均f1最大的index，对应最佳的iou阈值的位置
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index

    ## 根据i，选择p, r, f1的最佳iou阈值的数据，shape变为 class数
    p, r, f1 = p[:, i], r[:, i], f1[:, i]

    ## 根据r和p，反推该iou阈值时的每个class的tp和fp的数量
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives

    ## tp, fp, p, r, f1表示最佳置信度阈值下对应的值，shape都是class数
    ## ap的shape是class * 10，表示不同置信度阈值下的ap
    ## unique_classes.astype(int)是对应class的id
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)
