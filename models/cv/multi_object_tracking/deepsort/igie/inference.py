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
import argparse
import tvm
import torch
import torchvision
import numpy as np
from tvm import relay
from tqdm import tqdm

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

def get_dataloader(path, batch_size, num_workers):
    # data loader
    query_dir = os.path.join(path, "query")
    gallery_dir = os.path.join(path, "gallery")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    queryloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(query_dir, transform=transform),
        batch_size, shuffle=False, num_workers=num_workers
    )
    galleryloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
        batch_size, shuffle=True, num_workers=num_workers
    )

    return queryloader, galleryloader

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
        queryloader, galleryloader = get_dataloader(args.datasets, batch_size, args.num_workers)

        query_features = torch.tensor([]).float()
        query_labels = torch.tensor([]).long()
        gallery_features = torch.tensor([]).float()
        gallery_labels = torch.tensor([]).long()
        
        # run queryloader
        for inputs, labels in tqdm(queryloader):
            # Pad the last batch
            pad_batch = len(inputs) != batch_size
            if pad_batch:
                origin_size = len(inputs)
                inputs = np.resize(inputs, (batch_size, *inputs.shape[1:]))

            module.set_input('input', tvm.nd.array(inputs, device))
            
            module.run()

            features = torch.from_numpy(module.get_output(0).asnumpy())

            if pad_batch:
                features = features[:origin_size]

            query_features = torch.cat((query_features, features), dim=0)
            query_labels = torch.cat((query_labels, labels))

        # run galleryloader
        for inputs, labels in tqdm(galleryloader):
            # Pad the last batch
            pad_batch = len(inputs) != batch_size
            if pad_batch:
                origin_size = len(inputs)
                inputs = np.resize(inputs, (batch_size, *inputs.shape[1:]))

            module.set_input('input', tvm.nd.array(inputs, device))
            module.run()
            features = torch.from_numpy(module.get_output(0).asnumpy())

            if pad_batch:
                features = features[:origin_size]

            gallery_features = torch.cat((gallery_features, features), dim=0)
            gallery_labels = torch.cat((gallery_labels, labels))

        qf = query_features
        ql = query_labels
        gf = gallery_features
        gl = gallery_labels
        scores = qf.mm(gf.t())
        res = scores.topk(1, dim=1)[1][:,0]
        top1_correct = gl[res].eq(ql).sum().item()
        top1_acc = round(top1_correct / ql.size(0) * 100.0, 2)
        metricResult = {"metricResult": {"Acc": f"{top1_acc}%"}}
        print(metricResult)
        print(f"\n* Acc: {top1_acc} %")


if __name__ == "__main__":
    main()