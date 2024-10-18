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

import copy
import argparse
import numpy as np
from PIL import Image


import torch
import os 

from mmcv.image import imread
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope

from mmpose.apis import init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from tensorrt_common import create_engine_from_onnx,create_context,get_ixrt_output

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", 
                        type=str, 
                        required=True, 
                        help="the path of the model.")

    # parser.add_argument("--input_name", 
    #                     type=str, 
    #                     required=True, 
    #                     help="input name of the model.")
    
    parser.add_argument("--img_path", 
                        type=str, 
                        required=True, 
                        help="image path.")
            
    parser.add_argument("--conf",
                        type=float,
                        default=0.25,
                        help="confidence threshold.")
    
    parser.add_argument("--iou",
                        type=float,
                        default=0.65,
                        help="iou threshold.")
    
    parser.add_argument("--precision",
                        type=str,
                        choices=["fp32", "fp16", "int8"],
                        required=True,
                        help="model inference precision.")
    
    args = parser.parse_args()

    return args

def preprocess(model, img, bboxes=None, bbox_format="xyxy"):
    scope = model.cfg.get('default_scope', 'mmpose')
    
    if scope is not None:
        init_default_scope(scope)

    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # get bbox from the image size
    if isinstance(img, str):
        w, h = Image.open(img).size
    else:
        h, w = img.shape[:2]

    bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

    # construct batch data samples
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info['bbox'] = bbox[None]  # shape (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    data = pseudo_collate(data_list)

    return data


def main():
    args = parse_args()
    engine_file = args.model.replace(".onnx",".engine")
    create_engine_from_onnx(args.model,engine_file)
    
    engine, context = create_context(engine_file)
    
    

    model = init_model('rtmpose-m_8xb256-420e_coco-256x192.py')
    model.cfg.visualizer.radius = 3
    model.cfg.visualizer.alpha = 0.8
    model.cfg.visualizer.line_width = 1

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style="mmpose")
    
    outputs = []

    # get inputs
    inputs = preprocess(model, args.img_path)
    input_data = model.data_preprocessor(inputs, False)
    input_data = input_data['inputs'].cpu().numpy()
    
    outputs = get_ixrt_output(engine, context,input_data)

    preds = model.head.decode((torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])))

    if isinstance(preds, tuple):
        batch_pred_instances, batch_pred_fields = preds
    else:
        batch_pred_instances = preds
        batch_pred_fields = None

    batch_data_samples = model.add_pred_to_datasample(batch_pred_instances, batch_pred_fields, inputs['data_samples'])
    results = merge_data_samples(batch_data_samples)

    img = imread(args.img_path, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        out_file="./result.jpg")

    print("Results saved as result.jpg.")

if __name__ == "__main__":
    main()