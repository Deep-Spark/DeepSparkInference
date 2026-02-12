# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import sys
import argparse
import tvm
import torch
import numpy as np
from tvm import relay
from tqdm import tqdm

import timm
from timm.data import create_dataset, create_loader, resolve_data_config

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

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args

def get_topk_accuracy(pred, label):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
        
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    
    top1_acc = 0
    top5_acc = 0
    for idx in range(len(label)):
        label_value = label[idx]
        if label_value == torch.topk(pred[idx].float(), 1).indices.data:
            top1_acc += 1
            top5_acc += 1

        elif label_value in torch.topk(pred[idx].float(), 5).indices.data:
            top5_acc += 1
            
    return top1_acc, top5_acc

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
        # warm up
        for _ in range(args.warmup):
            module.run()

        # get dataloader
        model_name = 'mobilevit_s'
        model = timm.create_model(model_name, pretrained=False)
        config = resolve_data_config({}, model=model)
        dataset = create_dataset('imagenet', root=args.datasets, split='validation')
        dataloader = create_loader(
            dataset,
            input_size=(3, 224, 224),
            batch_size=batch_size,
            use_prefetcher=True,
            interpolation=config['interpolation'],
            mean=config['mean'],
            std=config['std'],
            crop_pct=config['crop_pct']
        )
        
        top1_acc = 0
        top5_acc = 0
        total_num = 0

        for image, label in tqdm(dataloader):
            
            image = image.cpu().numpy()
            label = label.cpu().numpy()

            # pad the last batch
            pad_batch = len(image) != batch_size
            
            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))

            module.set_input(args.input_name, tvm.nd.array(image, device))

            # run inference
            module.run()
            
            pred = module.get_output(0).asnumpy()

            if pad_batch:
                pred = pred[:origin_size]

            # get batch accuracy
            batch_top1_acc, batch_top5_acc = get_topk_accuracy(pred, label)

            top1_acc += batch_top1_acc
            top5_acc += batch_top5_acc
            total_num += batch_size

        result_stat = {}
        result_stat["acc@1"] = round(top1_acc / total_num * 100.0, 3)
        result_stat["acc@5"] = round(top5_acc / total_num * 100.0, 3)

        print(f"\n* Top1 acc: {result_stat['acc@1']} %, Top5 acc: {result_stat['acc@5']} %")

if __name__ == "__main__":
    main()