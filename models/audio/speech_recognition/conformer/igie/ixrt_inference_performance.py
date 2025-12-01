# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tvm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml
import time
import copy
import argparse
import pickle
import numpy as np

from tqdm.contrib import tqdm
from torch.utils.data import DataLoader

from wenet.file_utils import read_symbol_table
from wenet.dataset import Dataset

import tensorrt
from tensorrt import Dims
from common import create_engine_context, get_io_bindings,trtapi,setup_io_bindings
import pickle

import cuda.cuda as cuda
import cuda.cudart as cudart

from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

import tvm
from tvm import relay
from tvm.contrib import graph_executor

def get_args():
    parser = argparse.ArgumentParser(description="recognize with your model")
    parser.add_argument(
        "--infer_type",
        default="fp16",
        choices=["fp16", "int8"],
        help="inference type: fp16 or int8",
    )
    parser.add_argument("--warm_up", type=int, default=3, help="warm_up count")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--data_dir", required=True, help="test data directory")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="model for inference"
    )
    args = parser.parse_args()
    return args

def engine_init(engine):
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(engine, logger)
    
    return engine,context

def tensorrt_infer(engine,context, features, lengths):
    
    input_names=["input","seq_lengths"]
    output_names=["output"]
    input_idx = engine.get_binding_index(input_names[0])
    input_shape = features.shape    
    context.set_binding_shape(input_idx, Dims(input_shape))

    seq_lengths_idx = engine.get_binding_index(input_names[1])
    seq_lengths_shape = lengths.shape   
    context.set_binding_shape(seq_lengths_idx, Dims(seq_lengths_shape))
    
    inputs, outputs, allocations = setup_io_bindings(engine, context)
    pred_output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], features, features.nbytes)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuMemcpyHtoD(inputs[1]["allocation"], lengths, lengths.nbytes)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    context.execute_v2(allocations)
    err, = cuda.cuMemcpyDtoH(pred_output, outputs[0]["allocation"], outputs[0]["nbytes"])
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    return pred_output

def igie_infer(module, features, seq_lengths):
    start_time = time.time()
    module.set_input("input", features)
    module.set_input("seq_lengths", seq_lengths)
    module.run()
    out = module.get_output(0)
    eval_time = time.time() - start_time
    return out, eval_time

def igie_engine_init(engine_path):
    device = tvm.device("iluvatar", 0)
    lib = tvm.runtime.load_module(engine_path)
    module = graph_executor.GraphModule(lib["default"](device))
    # engine, context = module.engine, module.context
    return module

def main():
    args = get_args()

    # 读取配置文件
    config_fn = os.path.join(args.model_dir, "config.yaml")
    with open(config_fn, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    dataset_conf = copy.deepcopy(configs["dataset_conf"])
    dataset_conf["filter_conf"]["max_length"] = 102400
    dataset_conf["filter_conf"]["min_length"] = 0
    dataset_conf["filter_conf"]["token_max_length"] = 102400
    dataset_conf["filter_conf"]["token_min_length"] = 0
    dataset_conf["filter_conf"]["max_output_input_ratio"] = 102400
    dataset_conf["filter_conf"]["min_output_input_ratio"] = 0
    dataset_conf["speed_perturb"] = False
    dataset_conf["spec_aug"] = False
    dataset_conf["shuffle"] = False
    dataset_conf["sort"] = True
    dataset_conf["fbank_conf"]["dither"] = 0.0
    dataset_conf["batch_conf"]["batch_type"] = "static"
    dataset_conf["batch_conf"]["batch_size"] = args.batch_size

    # Load dict
    dict_fn = os.path.join(args.model_dir, "words.txt")
    char_dict = {}
    with open(dict_fn, "r", encoding="utf8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]

    print("*** 1. Prepare data ***")
    data_type = "raw"
    test_data_fn = os.path.join(args.data_dir, "data.list")
    symbol_table = read_symbol_table(dict_fn)
    test_dataset = Dataset(
        data_type, test_data_fn, symbol_table, dataset_conf, partition=False
    )
    
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    print("*** 2. Load IxRT engine ***")
    engine_path = os.path.join(args.model_dir, f"conformer_{args.infer_type}_trt.engine")
    # engine, context = engine_init(engine_path)
    module = igie_engine_init(engine_path)
    
    print("*** 3. Warm up ***")
    if args.warm_up > 0:
        for i in range(args.warm_up):
            module.run()

    print("*** 4. Inference ***")
    num_samples = 0
    results = []
    eval_time = 0.0
    for batch in test_data_loader:
        keys, feats, target, feats_lengths, target_lengths = batch
        feats = feats.cpu().numpy().astype(np.float16)
        feats_lengths = feats_lengths.cpu().numpy().astype(np.int32)
        num_samples += feats.shape[0]
        hyps, batch_eval_time = igie_infer(module, feats, feats_lengths)
        results.append([hyps, keys])
        eval_time += batch_eval_time

    QPS = num_samples / eval_time
    print(f"Recognize {num_samples} sentences, {QPS} sentences/s")
    target_qps = float(os.environ["Accuracy"])
    print("QPS: = ", QPS, "target QPS: ", target_qps)
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["QPS"] = round(QPS, 3)
    metricResult["metricResult"]["target QPS"] = round(target_qps, 3)
    print(metricResult)
    if QPS >= target_qps:
        print("pass!")
        exit()
    else:
        print("failed!")
        exit(10)


if __name__ == "__main__":
    main()
