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
import argparse
import tvm
import json
import torch
import numpy as np
from tvm import relay
from tqdm import tqdm

from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.data import converter
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.models.yolo.detect import DetectionValidator

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
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of workers used in pytorch dataloader.")
    
    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")
    
    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
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

class IGIEValidator(DetectionValidator):
    def __call__(self, engine, device, data):
        self.data = data
        self.stride = 32
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        self.init_metrics()

        # wram up
        for _ in range(3):
            engine.run()

        for batch in tqdm(self.dataloader):
            batch = self.preprocess(batch)

            imgs = batch['img']
            pad_batch = len(imgs) != self.args.batch
            if pad_batch:
                origin_size = len(imgs)
                imgs = np.resize(imgs, (self.args.batch, *imgs.shape[1:]))
            
            engine.set_input(0, tvm.nd.array(imgs, device))
            
            engine.run()

            outputs = engine.get_output(0).asnumpy()

            if pad_batch:
                outputs = outputs[:origin_size]
            
            outputs = torch.from_numpy(outputs)
            
            preds = self.postprocess([outputs])
            
            self.update_metrics(preds, batch)
        
        stats = self.get_stats()

        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                print(f'Saving {f.name} ...')
                json.dump(self.jdict, f)  # flatten and save

        stats = self.eval_json(stats)

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
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

def main():
    args = parse_args()

    batch_size = args.batchsize

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
        overrides = {'mode': 'val'}
        cfg_args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)

        cfg_args.batch = batch_size
        cfg_args.save_json = True

        data = {
            'path': Path(args.datasets),
            'val': os.path.join(args.datasets, 'val2017.txt'),
            'names': coco_classes
        }

        validator = IGIEValidator(args=cfg_args, save_dir=Path('.'))
        
        validator(module, device, data)
    
if __name__ == "__main__":
    main()