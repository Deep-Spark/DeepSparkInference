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
import sys
sys.path.insert(0, "YOLOv6")
import json
import argparse
import tvm
import torch
import numpy as np
from tvm import relay
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov6.core.evaler import Evaler
from yolov6.utils.events import NCOLS
from yolov6.utils.nms import non_max_suppression
from yolov6.data.data_load import create_dataloader


coco_classes = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class EvalerIGIE(Evaler):
    def eval_igie(self, engine, device, stride=32):
        self.stride = stride
        def init_data(dataloader, task):
            self.is_coco = self.data.get("is_coco", False)
            self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
            pad = 0.0 
            dataloader = create_dataloader(
                self.data[task], self.img_size, self.batch_size, self.stride, 
                check_labels=True, pad=pad, rect=False, data_dict=self.data, task=task)[0]
            return dataloader
        
        dataloader = init_data(None,'val')
        pred_results = []
        pbar = tqdm(dataloader, desc="Inferencing model in validation dataset.", ncols=NCOLS)
        
        for imgs, targes, paths, shapes in pbar:
            imgs = imgs.float()
            pad_batch = len(imgs) != self.batch_size
            if pad_batch:
                origin_size = len(imgs)
                imgs = np.resize(imgs, (self.batch_size, *imgs.shape[1:]))
            imgs /= 255.0

            engine.set_input(0, tvm.nd.array(imgs, device))
            
            engine.run()
            
            outputs = engine.get_output(0).asnumpy()

            if pad_batch:
                outputs = outputs[:origin_size]

            outputs = torch.from_numpy(outputs)
            outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=True)
            pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))
        
        return dataloader, pred_results
    
    def eval_igie_map(self, pred_results, dataloader, task):
        '''Evaluate models
            For task speed, this function only evaluates the speed of model and outputs inference time.
            For task val, this function evaluates the speed and mAP by pycocotools, and returns
            inference time and mAP value.
        '''
        if not self.do_coco_metric and self.do_pr_metric:
            return self.pr_metric_result
        print(f'\nEvaluating mAP by pycocotools.')
        if task != 'speed' and len(pred_results):
            if 'anno_path' in self.data:
                anno_json = self.data['anno_path']
            else:
                # generated coco format labels in dataset initialization
                task = 'val' if task == 'train' else task
                dataset_root = os.path.dirname(os.path.dirname(self.data[task]))
                base_name = os.path.basename(self.data[task])
                anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')
            pred_json = os.path.join(self.save_dir, "predictions.json")
            print(f'Saving {pred_json}...')
            with open(pred_json, 'w') as f:
                json.dump(pred_results, f)

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            cocoEval = COCOeval(anno, pred, 'bbox')
            if self.is_coco:
                imgIds = [int(os.path.basename(x).split(".")[0])
                            for x in dataloader.dataset.img_paths]
                cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            return cocoEval.stats

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--engine", 
                        type=str, 
                        required=True, 
                        help="igie engine path.")
    
    parser.add_argument("--batchsize",
                        type=int,
                        required=True, 
                        help="inference batch size.")
    
    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="datasets path.")

    parser.add_argument("--input_name", 
                        type=str, 
                        required=True, 
                        help="input name of the model.")
    
    parser.add_argument("--warmup", 
                        type=int, 
                        default=3, 
                        help="number of warmup before test.")           
        
    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")
    
    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
                        help="Model inference FPS target.")
    
    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    task = 'val'

    batch_size = args.batchsize
    data_path = os.path.join(args.datasets, "images", "val2017")
    label_path = os.path.join(args.datasets, "annotations", "instances_val2017.json")

    # create iluvatar target & device
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")    
    device = tvm.device(target.kind.name, 0)

    # load engine
    lib = tvm.runtime.load_module(args.engine)

    # create runtime from engine
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    # just run perf test
    if args.perf_only:
        ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)        
        prof_res = np.array(ftimer().results) * 1000 
        fps = batch_size * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
    else:
        # warm up
        for _ in range(args.warmup):
            module.run()

        data = {
            'task': 'val',
            'val': data_path,
            'anno_path': label_path,
            'names': coco_classes,
            'is_coco': True,
            'nc': 80
        }

        evaluator = EvalerIGIE(data, batch_size)
        dataloader, pred_results = evaluator.eval_igie(module, device)
        eval_result = evaluator.eval_igie_map(pred_results, dataloader, task)

if __name__ == "__main__":
    main()