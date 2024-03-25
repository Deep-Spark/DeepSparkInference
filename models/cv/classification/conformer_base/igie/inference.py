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

import sys
import argparse
import tvm
import torch
import torchvision
import numpy as np
from tvm import relay
from tqdm import tqdm
from timm.utils import accuracy
from torchvision import transforms

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

def get_dataloader(data_path, batch_size, num_workers):
    dataset = torchvision.datasets.ImageFolder(
        data_path,
        transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers)

    return dataloader

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
        dataloader = get_dataloader(args.datasets, batch_size, args.num_workers)
        
        top1_acc = []
        top5_acc = []

        for image, label in tqdm(dataloader):

            # pad the last batch
            pad_batch = len(image) != batch_size

            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))

            module.set_input(args.input_name, tvm.nd.array(image, device))

            # run inference
            module.run()
            
            acc1_head1 = module.get_output(0).asnumpy()

            acc1_head2 = module.get_output(1).asnumpy()

            pred = torch.from_numpy(acc1_head1 + acc1_head2)

            if pad_batch:
                pred = pred[:origin_size]

            top1_acc.append(accuracy(pred, label, topk=(1,))[0].data)

            top5_acc.append(accuracy(pred, label, topk=(5,))[0].data)

        result_stat = {}
        result_stat["acc@1"] = np.mean(top1_acc)
        result_stat["acc@5"] = np.mean(top5_acc)

        print(f"\n* Top1 acc: {result_stat['acc@1']:.3f} %, Top5 acc: {result_stat['acc@5']:.3f} %")


if __name__ == "__main__":
    main()