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
import numpy as np
from tvm import relay
from tqdm import tqdm
from mmdet.registry import RUNNERS
from mmengine.config import Config

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

        # runner config
        cfg = Config.fromfile("sabl-retinanet_r50_fpn_1x_coco.py")

        cfg.work_dir = "./workspace"
        cfg['test_dataloader']['batch_size'] = batch_size
        cfg['test_dataloader']['dataset']['data_root'] = args.datasets
        cfg['test_dataloader']['dataset']['data_prefix']['img'] = 'images/val2017/'
        cfg['test_evaluator']['ann_file'] = os.path.join(args.datasets, 'annotations/instances_val2017.json')
        cfg['log_level'] = 'ERROR'

        # build runner
        runner = RUNNERS.build(cfg)
    
        for data in tqdm(runner.test_dataloader):
            cls_score = []
            box_reg = []
            boxs = []
            
            input_data = runner.model.data_preprocessor(data, False)
            image = input_data['inputs'].cpu()
            pad_batch = len(image) != batch_size

            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))

            module.set_input("input", tvm.nd.array(image, device))

            module.run()
            
            for i in range(module.get_num_outputs()):
                output = module.get_output(i).asnumpy()

                if pad_batch:
                    output = output[:origin_size]

                output = torch.from_numpy(output)

                if output.shape[1] == 80:
                    cls_score.append(output)
                else:
                    boxs.append(output)

            for idx in range(0, len(boxs), 2):
                if boxs[idx].shape[-1] == boxs[idx+1].shape[-1]:
                    box_reg.append((boxs[idx], boxs[idx+1]))

            batch_img_metas = [
                data_samples.metainfo for data_samples in data['data_samples']
            ]  

            preds = runner.model.bbox_head.predict_by_feat(
                cls_score, box_reg, batch_img_metas=batch_img_metas, rescale=True
            )

            batch_data_samples = runner.model.add_pred_to_datasample(input_data['data_samples'], preds)

            runner.test_evaluator.process(data_samples=batch_data_samples, data_batch=data)

        metrics = runner.test_evaluator.evaluate(len(runner.test_dataloader.dataset))        
    

if __name__ == "__main__":
    main()
