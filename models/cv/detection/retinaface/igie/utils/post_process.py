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
import torch
import numpy as np
from .prior_box import PriorBox
from .box_utils import decode, decode_landm
from .py_cpu_nms import py_cpu_nms

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[10, 20], [32, 64], [128, 256]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

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

def clip_boxes_landm(landm, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(landm, torch.Tensor):  # faster individually
        landm[:, 0].clamp_(0, shape[1])  # x1
        landm[:, 1].clamp_(0, shape[0])  # y1
        landm[:, 2].clamp_(0, shape[1])  # x2
        landm[:, 3].clamp_(0, shape[0])  # y2
        landm[:, 4].clamp_(0, shape[1])  # x1
        landm[:, 5].clamp_(0, shape[0])  # y1
        landm[:, 6].clamp_(0, shape[1])  # x2
        landm[:, 7].clamp_(0, shape[0])  # y2
        landm[:, 8].clamp_(0, shape[1])  # x2
        landm[:, 9].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        landm[:, [0, 2,4,6,8]] = landm[:, [0, 2,4,6,8]].clip(0, shape[1])  # x1, x2
        landm[:, [1, 3,5,7,9]] = landm[:, [1, 3,5,7,9]].clip(0, shape[0])  # y1, y2

def scale_boxes_landm(img1_shape, landm, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    landm[:, [0, 2,4,6,8]] -= pad[0]  # x padding
    landm[:, [1, 3,5,7,9]] -= pad[1]  # y padding
    landm[:, :10] /= gain
   
    clip_boxes_landm(landm, img0_shape)
    return landm

def post_process(shapes, img_names, loc_bs, conf_bs, landms_bs, save_folder):
    max_size = 320
    confidence_threshold=0.02
    nms_threshold=0.4

    for idx, loc in enumerate(loc_bs):
        img_size=[320, 320]
        im_shape=list(shapes[idx][0])       #ori
        
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(320) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) >max_size:
            resize = float(max_size) / float(im_size_max)
        
        scale = torch.Tensor([img_size[1], img_size[0], img_size[1], img_size[0]])
        scale = scale.to('cpu')

        priorbox = PriorBox(cfg_mnet, image_size=(320, 320))
        priors = priorbox.forward()
        priors = priors.to('cpu')
        prior_data = priors.data
        
        boxes = decode(torch.from_numpy(loc_bs[idx]).data.squeeze(0).float(), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale
        boxes=scale_boxes([320, 320],boxes,im_shape,shapes[idx][1])
        boxes = boxes.cpu().numpy()
        scores = torch.from_numpy(conf_bs[idx]).squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(torch.from_numpy(landms_bs[idx]).data.squeeze(0), prior_data, cfg_mnet['variance'])
        img_size=[1,3,img_size[0],img_size[1]]


        scale1 = torch.Tensor([img_size[3], img_size[2], img_size[3], img_size[2],
                            img_size[3], img_size[2], img_size[3], img_size[2],
                            img_size[3], img_size[2]])
        scale1 = scale1.to('cpu')
        
        landms = landms * scale1
        landms=scale_boxes_landm([320, 320],landms,im_shape,shapes[idx][1])
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        
        # --------------------------------------------------------------------
        save_name = save_folder + img_names[idx][:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
