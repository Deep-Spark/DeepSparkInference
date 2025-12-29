# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import json
import argparse
import time
import tensorrt
from tensorrt import Dims
from cuda import cuda, cudart
import torch
import numpy as np
from tqdm import tqdm

from common import create_engine_context, setup_io_bindings

from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.data import converter
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.models.yolo.detect import DetectionValidator

coco_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
                40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
                50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
                70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_engine", 
                        type=str, 
                        required=True, 
                        help="ixrt engine path.")
    
    parser.add_argument("--bsz",
                        type=int,
                        required=True, 
                        help="inference batch size.")
    
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="inference size h,w",
    )
    
    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="datasets path.")
    
    parser.add_argument("--warm_up", 
                        type=int, 
                        default=3, 
                        help="number of warmup before test.")           
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of workers used in pytorch dataloader.")
    
    parser.add_argument("--acc_target",
                        type=float,
                        default=0.0,
                        help="Model inference Accuracy target.")
    
    parser.add_argument("--fps_target",
                        type=float,
                        default=0.0,
                        help="Model inference FPS target.")
    
    parser.add_argument("--conf",
                        type=float,
                        default=0.001,
                        help="confidence threshold.")
    
    parser.add_argument("--iou",
                        type=float,
                        default=0.65,
                        help="iou threshold.")

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args

class IxRT_Validator(DetectionValidator):
    def __call__(self, config, data):
        self.data = data
        self.stride = 32
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        self.init_metrics()
        
        total_num = 0

        input_name = "images"
        host_mem = tensorrt.IHostMemory
        logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        engine, context = create_engine_context(config.model_engine, logger)
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims((config.bsz,3,config.imgsz,config.imgsz)))
        inputs, outputs, allocations = setup_io_bindings(engine, context)
        
        if config.warm_up > 0:
            print("\nWarm Start.")
            for i in range(config.warm_up):
                context.execute_v2(allocations)
            print("Warm Done.")
        
        forward_time = 0.0
        num_samples = 0   

        e2e_start_time = time.time()
        for batch in tqdm(self.dataloader):
            batch = self.preprocess(batch)

            imgs = batch['img']
            pad_batch = len(imgs) != self.args.batch
            if pad_batch:
                origin_size = len(imgs)
                imgs = np.resize(imgs, (self.args.batch, *imgs.shape[1:]))
            
            batch_data = np.ascontiguousarray(imgs)
            data_shape = batch_data.shape
            
            cur_bsz_sample = batch_data.shape[0]
            num_samples += cur_bsz_sample

            # Set input
            input_idx = engine.get_binding_index(input_name)
            context.set_binding_shape(input_idx, Dims(data_shape))
            inputs, outputs, allocations = setup_io_bindings(engine, context)

            err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], batch_data, batch_data.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            # Prepare the output data
            output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
            
            
            start_time = time.time()
            context.execute_v2(allocations)
            end_time = time.time()
            forward_time += end_time - start_time
            
            err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)

            for alloc in allocations:
                if not alloc:
                    continue
                (err,) = cudart.cudaFree(alloc)
                assert err == cudart.cudaError_t.cudaSuccess   
                
            if pad_batch:
                output = output[:origin_size]
                
            outputs = torch.from_numpy(output)
            
            preds = self.postprocess([outputs])
            
            self.update_metrics(preds, batch)

        e2e_end_time = time.time()
        if config.perf_only:
            fps = num_samples / forward_time
            return fps
        else:
            stats = self.get_stats()

            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    print(f'Saving {f.name} ...')
                    json.dump(self.jdict, f)  # flatten and save

            stats = self.eval_json(stats)

            end2end_time = e2e_end_time - e2e_start_time
            print(F"E2E time : {end2end_time:.3f} seconds")

            return stats

    def init_metrics(self):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')  # is COCO
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = self.data['names']
        self.nc = len(self.names)
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(nc=80)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

def main():
    config = parse_args()

    batch_size = config.bsz

    overrides = {'mode': 'val'}
    cfg_args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)

    cfg_args.batch = batch_size
    cfg_args.save_json = True

    data = {
        'path': Path(config.datasets),
        'val': os.path.join(config.datasets, 'val2017.txt'),
        'names': coco_classes
    }

    validator = IxRT_Validator(args=cfg_args, save_dir=Path('.'))
    
    if config.perf_only:
        fps = validator(config, data)
        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {config.fps_target}")
    else:
        stats = validator(config, data)
        
    
if __name__ == "__main__":
    main()