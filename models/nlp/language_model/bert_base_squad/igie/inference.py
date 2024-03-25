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

    parser.add_argument("--input_name", 
                        type=str,
                        nargs='+', 
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

def main():
    args = parse_args()

    batch_size = args.batchsize
    seq_length = args.seqlen
    checkpoint = "csarron/bert-base-uncased-squad-v1"

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
        _, examples, features, dataloader = get_squad_dataloader(args.datasets, batch_size, seq_length, checkpoint)
        
        all_results = []

        for batch in tqdm(dataloader):
            input_ids = batch[0].numpy()
            attention_mask = batch[1].numpy()
            token_type_ids = batch[2].numpy()

            module.set_input(args.input_name[0], tvm.nd.array(input_ids, device))
            module.set_input(args.input_name[1], tvm.nd.array(attention_mask, device))
            module.set_input(args.input_name[2], tvm.nd.array(token_type_ids, device))
            example_indices = batch[3]

            # run inference
            module.run()
            
            start_logits = module.get_output(0).asnumpy()
            end_logits = module.get_output(1).asnumpy()

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

if __name__ == "__main__":
    main()