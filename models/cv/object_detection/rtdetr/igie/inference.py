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
import torch
import numpy as np
from tvm import relay
from tqdm import tqdm

from utils import COCO2017Dataset, COCO2017Evaluator

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

def get_dataloader(data_path, label_path, batch_size, num_workers):

    dataset = COCO2017Dataset(data_path, label_path, image_size=(640, 640))

    dataloader = torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    collate_fn=dataset.collate_fn)
    return dataloader

def main():
    args = parse_args()

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

        # get dataloader
        dataloader = get_dataloader(data_path, label_path, batch_size, args.num_workers)

        # get evaluator
        evaluator = COCO2017Evaluator(label_path=label_path,
                                    conf_thres=args.conf,
                                    iou_thres=args.iou,
                                    image_size=640)
        
        for all_inputs in tqdm(dataloader):
            image = all_inputs[0]
            pred = None

            pad_batch = len(image) != batch_size
            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))
            
            module.set_input(args.input_name, tvm.nd.array(image, device))
            
            module.run()

            for i in range(module.get_num_outputs()):
                output = module.get_output(i).asnumpy()

                if pad_batch:
                    output = output[:origin_size]
                
                if pred is None:
                    pred = torch.from_numpy(output)
                else:
                    pred = torch.cat((pred, torch.from_numpy(output)), dim=-1)
            
            evaluator.evaluate(pred, all_inputs)
        
    
        evaluator.summary()
    
if __name__ == "__main__":
    main()