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
import yaml
import logging
import numpy as np
from tvm import relay
from tqdm import tqdm

import paddle
from paddleocr import ppocr
from ppocr.data import build_dataloader
from ppocr.utils.logging import get_logger
from ppocr.postprocess import build_post_process

class VQASerTokenMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        preds, labels = preds
        self.pred_list.extend(preds)
        self.gt_list.extend(labels)

    def get_metric(self):
        from seqeval.metrics import f1_score, precision_score, recall_score
        metrics = {
            "precision": precision_score(self.gt_list, self.pred_list),
            "recall": recall_score(self.gt_list, self.pred_list),
            "hmean": f1_score(self.gt_list, self.pred_list),
        }
        self.reset()
        return metrics

    def reset(self):
        self.pred_list = []
        self.gt_list = []

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

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    logger = get_logger(log_level=logging.INFO)
    
    input_names = args.input_name.split(",")
    config = yaml.load(open("ser_vi_layoutxlm_xfund_zh.yml", 'rb'), Loader=yaml.Loader)
    config['Eval']['loader']['batch_size_per_card'] = args.batchsize
    config['Eval']['dataset']['data_dir'] = os.path.join(args.datasets, "zh_val/image")
    config['Eval']['dataset']['label_file_list'] = os.path.join(args.datasets, "zh_val/val.json")
    config['Eval']['dataset']['transforms'][1]['VQATokenLabelEncode']['class_path'] = os.path.join(args.datasets, "class_list_xfun.txt")
    config['PostProcess']['class_path'] = os.path.join(args.datasets, "class_list_xfun.txt")

    # build dataloader
    config['Eval']['loader']['drop_last'] = True
    valid_dataloder = build_dataloader(config, 'Eval', paddle.set_device("cpu"), logger)

    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # build metric
    eval_class = eval('VQASerTokenMetric')()

    # creat target and device
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
        fps = args.batchsize * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
    else:
        # warm up
        for _ in range(args.warmup):
            module.run()

        for batch in tqdm(valid_dataloder):
            
            for idx, input_name in enumerate(input_names):
                module.set_input(input_name, tvm.nd.array(batch[idx], device))

            module.run()

            outputs = module.get_output(0).asnumpy()
            outputs = paddle.to_tensor(outputs)

            batch_numpy = []
            for item in batch:
                batch_numpy.append(item.numpy())
            
            post_result = post_process_class((outputs), batch_numpy)
            eval_class(post_result, batch_numpy)

        metric = eval_class.get_metric()
        metricResult = {"metricResult": {"hmean": metric["hmean"]}}
        print(metricResult)
        print(metric)

if __name__ == "__main__":
    main()
