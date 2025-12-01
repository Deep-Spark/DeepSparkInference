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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import yaml
import copy
import numpy as np

from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
from wenet.file_utils import read_symbol_table
from wenet.dataset import Dataset
from tools.compute_cer import Calculator, characterize, normalize, default_cluster
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



def ixrt_infer(module, input, seq_lengths):
    module.set_input(key="input", value=input)
    module.set_input(key="seq_lengths", value=seq_lengths)
    module.run()
    out = module.get_output()
    return out[0]


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


def engine_init(engine):
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(engine, logger)
    
    return engine,context


def igie_infer(module, features, seq_lengths):
    module.set_input("input", features)
    module.set_input("seq_lengths", seq_lengths)
    module.run()
    out = module.get_output(0)
    return out

def igie_engine_init(engine_path):
    device = tvm.device("iluvatar", 0)
    lib = tvm.runtime.load_module(engine_path)
    module = graph_executor.GraphModule(lib["default"](device))
    # engine, context = module.engine, module.context
    return module



def calculate_cer(data, reference_data):
    calculator = Calculator()
    tochar = True
    split = None
    case_sensitive = False
    ignore_words = set()
    rec_set = {}
    for line in data:
        if tochar:
            array = characterize(line)
        else:
            array = line.strip().split()
        if len(array) == 0:
            continue
        fid = array[0]
        rec_set[fid] = normalize(array[1:], ignore_words, case_sensitive, split)

    default_clusters = {}
    default_words = {}
    for line in reference_data:
        if tochar:
            array = characterize(line)
        else:
            array = line.strip().split()
        if len(array) == 0:
            continue
        fid = array[0]
        if fid not in rec_set:
            continue
        lab = normalize(array[1:], ignore_words, case_sensitive, split)
        rec = rec_set[fid]

        for word in rec + lab:
            if word not in default_words:
                default_cluster_name = default_cluster(word)
                if default_cluster_name not in default_clusters:
                    default_clusters[default_cluster_name] = {}
                if word not in default_clusters[default_cluster_name]:
                    default_clusters[default_cluster_name][word] = 1
                default_words[word] = default_cluster_name
        result = calculator.calculate(lab, rec)

    result = calculator.overall()
    cer = float(result["ins"] + result["sub"] + result["del"]) / result["all"]
    corr = result["cor"] / result["all"]

    return cer, corr


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
    eos = len(char_dict) - 1

    print("*** 1. Prepare data ***")
    data_type = "raw"
    test_data_fn = os.path.join(args.data_dir, "data.list")
    symbol_table = read_symbol_table(dict_fn)
    test_dataset = Dataset(
        data_type, test_data_fn, symbol_table, dataset_conf, partition=False
    )
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    print("*** 2. Load engine ***")
    engine_path = os.path.join(args.model_dir, f"conformer_{args.infer_type}_trt.engine")
    module = igie_engine_init(engine_path)
    
    print("*** 3. Warm up ***")
    if args.warm_up > 0:
        for i in range(args.warm_up):
            module.run()

    results = []
    for batch in test_data_loader:
        keys, feats, target, feats_lengths, target_lengths = batch
        feats = feats.cpu().numpy().astype(np.float16)
        feats_lengths = feats_lengths.cpu().numpy().astype(np.int32)
        hyps = igie_infer(module, feats, feats_lengths)
        for i, key in enumerate(keys):
            line = f"{key} "
            for w in hyps[i]:
                if w == eos:
                    break
                line += char_dict[w]
            results.append(line)

    # 3. 计算 CER
    reference_file = os.path.join(args.data_dir, "text")
    reference_data = []
    for line in open(reference_file, "r", encoding="utf-8"):
        reference_data.append(line)

    cer, corr = calculate_cer(results, reference_data)

    target_cer = float(os.environ["Accuracy"])
    print("CER: ", cer, "target CER: ", target_cer)
    if cer <= target_cer:
        print("pass!")
        exit()
    else:
        print("failed!")
        exit(1)


if __name__ == "__main__":
    main()
