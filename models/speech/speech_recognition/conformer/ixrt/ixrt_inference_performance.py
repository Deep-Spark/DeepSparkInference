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
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import yaml
import copy
import torch
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

import pycuda.autoinit
import pycuda.driver as cuda

from utils import make_pad_mask, RelPositionalEncoding
from postprocess import ctc_greedy_search


rel_positional_encoding = RelPositionalEncoding(256, 0.1)


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


def tensorrt_infer(engine, context, all_inputs):

    input_names = ["input", "mask", "pos_emb"]
    output_names = ["output"]

    for input_name, input_data in zip(input_names, all_inputs):
        input_idx = engine.get_binding_index(input_name)
        input_shape = input_data.shape
        context.set_binding_shape(input_idx, Dims(input_shape))

    inputs, outputs, allocations = setup_io_bindings(engine, context)
    pred_output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

    for i, input_data in enumerate(all_inputs):
        cuda.memcpy_htod(inputs[i]["allocation"], input_data)

    context.execute_v2(allocations)
    cuda.memcpy_dtoh(pred_output, outputs[0]["allocation"])
    return pred_output


def engine_init(engine):
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(engine, logger)

    return engine,context


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

    data_type = "raw"
    test_data_fn = os.path.join(args.data_dir, "data.list")
    symbol_table = read_symbol_table(dict_fn)
    test_dataset = Dataset(
        data_type, test_data_fn, symbol_table, dataset_conf, partition=False
    )
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    data_path_pkl = os.path.join(args.data_dir, f"aishell_test_data_bs{args.batch_size}.pkl")

    print("*** 1. Prepare data ***")
    if not os.path.isfile(data_path_pkl):
        eval_samples = []
        max_batch_size = -1
        max_feature_length = -1
        for batch in test_data_loader:
            keys, feats, target, feats_lengths, target_lengths = batch
            max_feature_length = max(max_feature_length, feats.size(1))
            max_batch_size = max(max_batch_size, feats.size(0))
            eval_samples.append(
                [
                    keys,
                    feats.cpu().numpy().astype(np.float16),
                    feats_lengths.cpu().numpy().astype(np.int32),
                ]
            )
        with open(data_path_pkl, "wb") as f:
            pickle.dump(
                [
                    eval_samples,
                    max_batch_size,
                    max_feature_length
                ],
                f,
            )
    else:
        print(f"load data from tmp: {data_path_pkl}")
        with open(data_path_pkl, "rb") as f:
            (
                eval_samples,
                max_batch_size,
                max_feature_length
            ) = pickle.load(f)
    print(
        f"dataset max shape: batch_size: {max_batch_size}, feat_length: {max_feature_length}"
    )

    print("*** 2. Load engine ***")
    engine_path = os.path.join(args.model_dir, f"conformer_encoder_fusion.engine")
    engine, context = engine_init(engine_path)

    print("*** 3. Warm up ***")
    if args.warm_up > 0:
        for i in range(args.warm_up):
            feats_tmp = np.ones((args.batch_size, 1500, 80)).astype(np.float32)
            feats_lengths_tmp = np.ones((args.batch_size)).astype(np.int32) * 1500
            mask_tmp = make_pad_mask(feats_lengths_tmp, 1500)
            mask_len_tmp = mask_tmp.shape[-1]
            pos_emb_tmp = rel_positional_encoding(mask_len_tmp).numpy()
            all_inputs = [feats_tmp, mask_tmp, pos_emb_tmp]
            tensorrt_infer(engine, context, all_inputs)

    print("*** 4. Inference ***")
    start_time = time.time()
    num_samples = 0
    results = []
    for keys, feats, feats_lengths in tqdm(eval_samples):
        b, seq_len, feat = feats.shape
        num_samples += b
        inputs = feats.astype(np.float32)
        mask = make_pad_mask(feats_lengths, seq_len)
        mask_len = mask.shape[-1]
        pos_emb = rel_positional_encoding(mask_len).numpy()

        all_inputs = [inputs, mask, pos_emb]
        hyps = tensorrt_infer(
            engine,
            context,
            all_inputs
        )

    eval_time = time.time() - start_time

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
        exit(1)


if __name__ == "__main__":
    main()
