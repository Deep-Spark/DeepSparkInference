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

from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, ListDataset
from bert4torch.tokenizers import Tokenizer

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

class NerDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                d = ['']
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                D.append(d)
        return D

def trans_entity2tuple(scores, categories_id2label):
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:] == entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids

def evaluate(args, input_token_ids, batch_labels, outputs, module, device, categories_id2label):
    all_scores = []
    all_labels = []
    
    cnt = 0
    num_samples = 0

    device.sync()
    start_time = time.time()

    for batch_token_ids in tqdm(input_token_ids):
        num_samples += batch_token_ids.shape[0]
        scores = module.inference(batch_token_ids).numpy()
        outputs[cnt] = scores
        all_scores.append(outputs[cnt])
        cnt += 1
    device.sync()
    infer_time = time.time() - start_time
    all_labels = batch_labels

    metricResult = {"metricResult": {}}
    if args.perf_only:
        run_time = infer_time / num_samples
        fps = 1.0 / run_time
        print(f"\n* Mean inference time: {run_time:.5f} ms, Mean fps: {fps:.3f}")
        metricResult["metricResult"]["Mean inference time"] = run_time
        metricResult["metricResult"]["Mean fps"] = fps
    else:
        print(f"\ncompute evaluation metrics...")
        X, Y, Z = 1e-10, 1e-10, 1e-10
        X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
        def pad_scores(scores):
            for i in range(len(scores)):
                score = scores[i]
                score = np.pad(score, ((0, 0), (0, args.seqlen - score.shape[1])))
                scores[i] = score
            return scores
        all_scores = pad_scores(all_scores)

        for scores, label in zip(all_scores, all_labels):
            attention_mask = label.gt(0)
            scores = torch.from_numpy(scores).cuda()
            if scores.shape[0] != label.shape[0]:
                scores = scores[:label.shape[0]]
            # token粒度
            X += (scores.eq(label) * attention_mask).sum().item()
            Y += scores.gt(0).sum().item()
            Z += label.gt(0).sum().item()

            # entity粒度
            entity_pred = trans_entity2tuple(scores, categories_id2label)
            entity_true = trans_entity2tuple(label, categories_id2label)
            X2 += len(entity_pred.intersection(entity_true))
            Y2 += len(entity_pred)
            Z2 += len(entity_true)

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
        print(f'\n[val-token  level] f1: {f1:.3f}, p: {precision:.3f} r: {recall:.3f}')
        print(f'\n[val-entity level] f1: {f2:.3f}, p: {precision2:.3f} r: {recall2:.3f}')

        metricResult["metricResult"]["val-token f1"] = f1
        metricResult["metricResult"]["val-entity f1"] = f2
    print(metricResult)

def main():
    args = parse_args()

    dict_path = "test/vocab.txt"
    categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
    categories_id2label = {i: k for i, k in enumerate(categories)}
    categories_label2id = {k: i for i, k in enumerate(categories)}
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    batch_size = args.batchsize
    seq_length = args.seqlen
    datasets = os.path.join(args.datasets, "example.dev")

    def collate_fn(batch):
        batch_token_ids, batch_labels = [], []
        length = 0
        for d in batch:
            tokens = tokenizer.tokenize(d[0], maxlen=seq_length)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            length = max(length, len(token_ids))
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories_label2id['B-'+label]
                    labels[start + 1:end + 1] = categories_label2id['I-'+label]
            batch_token_ids.append(token_ids)
            batch_labels.append(labels)

        if len(batch_token_ids) != batch_size:
            for i in range(batch_size - len(batch_token_ids)):
                batch_token_ids.append([0] * seq_length)
        if length % 4 != 0:
            length = 4 - length % 4 + length
        batch_token_ids = torch.tensor(sequence_padding(
            batch_token_ids, length=seq_length), dtype=torch.long)
        batch_labels = torch.tensor(sequence_padding(
            batch_labels, length=seq_length), dtype=torch.long)
        
        return batch_token_ids, batch_labels

    # create iluvatar target & device
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")    
    device = tvm.device(target.kind.name, 0)

    engine_path = args.engine

    if os.path.isdir(engine_path):
        engine_path = os.path.join(engine_path, f"bert_base_ner_int8_b{batch_size}_seq{seq_length}.so")

    # load engine
    module = relay.frontend.Bert.load_module(batch_size, seq_length, engine_path)

    all_token_ids = []
    all_labels = []
    outputs = []

    valid_dataloader = DataLoader(NerDataset(datasets), batch_size=batch_size,collate_fn=collate_fn)
    
    total_batch = 0
    
    for batch_token_ids, batch_labels in valid_dataloader:
        token_ids = batch_token_ids.numpy().astype("int32")
        all_token_ids.append(token_ids)
        all_labels.append(batch_labels.to("cuda:0"))
        total_batch += batch_labels.shape[0]

    batches = []
    for token_ids in all_token_ids:
        batch_length = np.sum(token_ids != 0, axis=-1).max()    
        batch_length = args.seqlen
        token_ids = token_ids[:, :batch_length]
        batches.append(tvm.nd.array(token_ids))
        outputs.append(tvm.nd.empty((batch_size, batch_length), dtype="int32"))
    all_token_ids = batches

    evaluate(args, all_token_ids, all_labels, outputs, module, device, categories_id2label)

if __name__ == "__main__":
    main()
