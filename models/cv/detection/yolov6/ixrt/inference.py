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
import time
import tensorrt
from tensorrt import Dims
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import numpy as np
from tqdm import tqdm

from common import create_engine_context, setup_io_bindings

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

class EvalerIXRT(Evaler):
    def eval_ixrt(self, args, stride=32):
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
        
        input_name = "input"
        host_mem = tensorrt.IHostMemory
        logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        engine, context = create_engine_context(args.model_engine, logger)
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims((args.bsz,3,args.imgsz,args.imgsz)))
        inputs, outputs, allocations = setup_io_bindings(engine, context)
        
        if args.warm_up > 0:
            print("\nWarm Start.")
            for i in range(args.warm_up):
                context.execute_v2(allocations)
            print("Warm Done.")
        
        pbar = tqdm(dataloader, desc="Inferencing model in validation dataset.", ncols=NCOLS)
        
        forward_time = 0.0
        num_samples = 0   
        for imgs, targes, paths, shapes in pbar:
            imgs = imgs.float()
            pad_batch = len(imgs) != self.batch_size
            if pad_batch:
                origin_size = len(imgs)
                imgs = np.resize(imgs, (self.batch_size, *imgs.shape[1:]))
            imgs /= 255.0
            # print(imgs.shape)
            batch_data = np.ascontiguousarray(imgs)
            data_shape = batch_data.shape
            
            cur_bsz_sample = batch_data.shape[0]
            num_samples += cur_bsz_sample

            # Set input
            input_idx = engine.get_binding_index(input_name)
            context.set_binding_shape(input_idx, Dims(data_shape))
            inputs, outputs, allocations = setup_io_bindings(engine, context)

            cuda.memcpy_htod(inputs[0]["allocation"], batch_data)
            # Prepare the output data
            output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
            
            
            start_time = time.time()
            context.execute_v2(allocations)
            end_time = time.time()
            forward_time += end_time - start_time
            
            cuda.memcpy_dtoh(output, outputs[0]["allocation"])
            
            if not args.perf_only:
                if pad_batch:
                    output = output[:origin_size]

                outputs = torch.from_numpy(output)
                outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=True)
                pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))
        if args.perf_only:       
            fps = num_samples / forward_time
            return fps
        else:
            return dataloader, pred_results
    
    def eval_ixrt_map(self, pred_results, dataloader, task):
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
        else:
            print("pred_results is none")
            return None

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_engine",
        type=str,
        default="",
        help="model engine path",
    )
    
    parser.add_argument("--bsz", type=int, default=32, help="test batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=608,
        help="inference size h,w",
    )
    
    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="datasets path.")
    
    parser.add_argument("--warm_up", type=int, default=3, help="warm_up count")          
        
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

    batch_size = args.bsz
    data_path = os.path.join(args.datasets, "images", "val2017")
    label_path = os.path.join(args.datasets, "annotations", "instances_val2017.json")
        

    data = {
        'task': 'val',
        'val': data_path,
        'anno_path': label_path,
        'names': coco_classes,
        'is_coco': True,
        'nc': 80
    }

    evaluator = EvalerIXRT(data, batch_size)
    
    if args.perf_only:
        fps = evaluator.eval_ixrt(args)
        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {args.fps_target}")
    else:
        dataloader, pred_results = evaluator.eval_ixrt(args)
        eval_result = evaluator.eval_ixrt_map(pred_results, dataloader, task)
        map, map50 = eval_result[:2]
        print("MAP@0.5 : ", map50)
        print(f"Accuracy Check : Test {map50} >= target {args.acc_target}")
        if map50 >= args.acc_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)

if __name__ == "__main__":
    main()