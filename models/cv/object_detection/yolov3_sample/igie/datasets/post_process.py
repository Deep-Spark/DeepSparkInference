import cv2
import math
import numpy as np

from .common import letterbox, scale_boxes, clip_boxes

def get_post_process(data_process_type):
    if data_process_type == "yolov5":
        return Yolov5Postprocess
    elif data_process_type == "yolov3":
        return Yolov3Postprocess
    elif data_process_type == "yolox":
        return YoloxPostprocess
    return None

def Yolov3Postprocess(
    ori_img_shape,
    imgsz,
    box_datas,
    box_nums,
    sample_num,
    max_det=1000,
):
    all_box = []
    data_offset = 0

    box_datas = box_datas.flatten()
    box_nums = box_nums.flatten()

    for i in range(sample_num):
        box_num = box_nums[i]
        if box_num == 0:
            boxes = None
        else:
            cur_box = box_datas[data_offset : data_offset + box_num * 6].reshape(-1, 6)
            boxes = scale_boxes(
                (imgsz[0], imgsz[1]),
                cur_box,
                (ori_img_shape[0][i], ori_img_shape[1][i]),
                use_letterbox=False
            )
            # xyxy2xywh
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

        all_box.append(boxes)
        data_offset += max_det * 6

    return all_box

def Yolov5Postprocess(
    ori_img_shape,
    imgsz,
    box_datas,
    box_nums,
    sample_num,
    max_det=1000,
):
    all_box = []
    data_offset = 0

    box_datas = box_datas.flatten()
    box_nums = box_nums.flatten()

    for i in range(sample_num):
        box_num = box_nums[i]
        if box_num == 0:
            boxes = None
        else:
            cur_box = box_datas[data_offset : data_offset + box_num * 6].reshape(-1, 6)
            boxes = scale_boxes(
                (imgsz[0], imgsz[1]),
                cur_box,
                (ori_img_shape[0][i], ori_img_shape[1][i]),
                use_letterbox=True
            )
            # xyxy2xywh
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

        all_box.append(boxes)
        data_offset += max_det * 6

    return all_box

def YoloxPostprocess(
    ori_img_shape,
    imgsz,
    box_datas,
    box_nums,
    sample_num,
    max_det=1000,
):
    all_box = []
    data_offset = 0
    box_datas = box_datas.flatten()
    box_nums = box_nums.flatten()

    for i in range(sample_num):
        box_num = box_nums[i]
        if box_num == 0:
            boxes = None
        else:
            boxes = box_datas[data_offset : data_offset + box_num * 6].reshape(-1, 6)
            r = min(imgsz[0]/ori_img_shape[0][i], imgsz[1]/ori_img_shape[1][i])
            boxes[:, :4] /= r
            # xyxy2xywh
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            clip_boxes(boxes, (ori_img_shape[0][i], ori_img_shape[1][i]))

        all_box.append(boxes)
        data_offset += max_det * 6

    return all_box