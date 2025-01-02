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

import time

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from torch.utils.data import DataLoader, SequentialSampler
from transformers import squad_convert_examples_to_features, AutoTokenizer
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

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

    parser.add_argument("--seqlen",
                        type=int,
                        default=256,
                        help="inference sequence length.")
    
    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="datasets path.")
    
    parser.add_argument("--precision",
                        type=str,
                        choices=["fp32", "fp16", "int8"],
                        required=True,
                        help="model inference precision.")

    parser.add_argument("--input_name", 
                        type=str,
                        nargs='+', 
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

def load_and_cache_examples(dataset_dir,
                            checkpoint,
                            tokenizer,
                            evaluate=True,
                            output_examples=True,
                            use_v1=True,
                            max_seq_length=512,
                            doc_stride=128,
                            max_query_length=64
                            ):

    cached_features_file = os.path.join(dataset_dir, "cached_{}_{}_{}".format("squad_v1.1" if use_v1 else "squad_v2.0", "dev" if evaluate else "train", checkpoint.split("/")[-1]))

    if os.path.exists(cached_features_file):
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (features_and_dataset["features"], features_and_dataset["dataset"], features_and_dataset["examples"])
    else:
        assert os.path.exists(dataset_dir)

        if use_v1: # squadv1.1
            processor = SquadV1Processor()
        else: # squadv2.0
            processor = SquadV2Processor()
            
        if evaluate:
            examples = processor.get_dev_examples(dataset_dir)
        else:
            examples = processor.get_train_examples(dataset_dir)
        
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
        )
        
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)


    if output_examples:
        return dataset, examples, features
    return dataset

def get_squad_dataloader(dataset_dir, batch_size, seq_length, checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    dataset, examples, features = load_and_cache_examples(dataset_dir, checkpoint, tokenizer, max_seq_length=seq_length)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, drop_last=True)
    
    return dataset, examples, features, eval_dataloader

def run_engine(args, module, data, device):
    if args.precision != "int8":
        input_ids = data[0].numpy()
        attention_mask = data[1].numpy()
        token_type_ids = data[2].numpy()

        module.set_input(args.input_name[0], tvm.nd.array(input_ids, device))
        module.set_input(args.input_name[1], tvm.nd.array(attention_mask, device))
        module.set_input(args.input_name[2], tvm.nd.array(token_type_ids, device))

        # run inference
        module.run()
        
        start_logits = module.get_output(0).asnumpy()
        end_logits = module.get_output(1).asnumpy()
    
    else:
        token_ids = data[0].numpy().astype("int32")
        logits = module.inference(token_ids).numpy()

        start_logits = logits[:,:,0]
        end_logits = logits[:,:,1]

    return start_logits, end_logits

def main():
    args = parse_args()

    batch_size = args.batchsize
    seq_length = args.seqlen
    checkpoint = "neuralmagic/bert-large-uncased-finetuned-squadv1"

    # create iluvatar target & device
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")    
    device = tvm.device(target.kind.name, 0)

    engine_path = args.engine

    if os.path.isdir(engine_path):
        engine_path = os.path.join(engine_path, f"bert_large_squad_int8_b{batch_size}_seq{seq_length}.so")

    _, examples, features, dataloader = get_squad_dataloader(args.datasets, batch_size, seq_length, checkpoint)

    # load engine
    if args.precision == "int8":
        module = relay.frontend.Bert.load_module(batch_size, seq_length, engine_path)
    else:
        lib = tvm.runtime.load_module(engine_path)
        # create runtime from engine
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    metricResult = {"metricResult": {}}
    # just run perf test
    if args.perf_only:
        if args.precision != "int8":
            ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)        
            prof_res = np.array(ftimer().results) * 1000
            run_time = np.mean(prof_res)
            fps = batch_size * 1000 / np.mean(prof_res)
            print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
        else:
            num_samples = len(dataloader.dataset)

            device.sync()
            start_time = time.time()
            for batch in dataloader:
                start_logits, end_logits = run_engine(args, module, batch, device)
            device.sync()

            infer_time = time.time() - start_time
            run_time = infer_time / num_samples
            fps = 1.0 / run_time
            print(f"\n* Mean inference time: {run_time:.3f} ms, Mean fps: {fps:.3f}")

        metricResult["metricResult"]["Mean inference time"] = run_time
        metricResult["metricResult"]["Mean fps"] = fps
    else:
        # warm up
        if args.precision != "int8":
            for _ in range(args.warmup):
                module.run()

        all_results = []

        for batch in tqdm(dataloader):
            example_indices = batch[3]

            start_logits, end_logits = run_engine(args, module, batch, device)

            for idx, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = SquadResult(unique_id, start_logits[idx], end_logits[idx])
                all_results.append(result)

        features = features[:len(all_results)]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        predictions = compute_predictions_logits(examples, features, all_results, 20, 30, True,
                                            None, None, None, False, False, 0.0, tokenizer)
        results = squad_evaluate(examples, predictions)

        print(f"\n F1 Score: {results['f1']:.3f}")
        metricResult["metricResult"]["F1 Score"] = results['f1']
    print(metricResult)

if __name__ == "__main__":
    main()