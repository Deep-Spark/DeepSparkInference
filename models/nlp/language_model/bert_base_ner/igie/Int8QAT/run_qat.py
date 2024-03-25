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

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from bert4torch.layers import CRF
from bert4torch.models import BaseModel, build_transformer_model
from bert4torch.snippets import Callback, ListDataset, seed_everything, sequence_padding
from bert4torch.tokenizers import Tokenizer
from ls_hf_transformer_layer import inject_ls_layer

maxlen = 256
batch_size = 16
categories = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", 
                        type=str, 
                        required=True, 
                        help="pytorch weights dir.")
    
    parser.add_argument("--datasets_dir", 
                        type=str, 
                        required=True, 
                        help="datasets dir.")
    
    args = parser.parse_args()

    return args

class quan_model_config:
    module_type = 2
    quant_mode = "qat"
    enable_quant = True


class quan_train_config:
    fp16 = False
    local_rank = -1


class quant_bert_config:
    pass

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding="utf-8") as f:
            f = f.read()
            for l in f.split("\n\n"):
                if not l:
                    continue
                d = [""]
                for i, c in enumerate(l.split("\n")):
                    char, flag = c.split(" ")
                    d[0] += char
                    if flag[0] == "B":
                        d.append([i, i, flag[2:]])
                    elif flag[0] == "I":
                        d[-1][1] = i
                D.append(d)
        return D

class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            segment_vocab_size=0,
        )
        self.fc = nn.Linear(768, len(categories))
        self.crf = CRF(len(categories))

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            emission_score, attention_mask = self.forward(token_ids)
            best_path = self.crf.decode(
                emission_score, attention_mask
            )  # [btz, seq_len]
        return best_path

class Loss(nn.Module):
    def forward(self, outputs, labels):
        return model.crf(*outputs, labels)

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories_label2id["B-" + label]
                labels[start + 1 : end + 1] = categories_label2id["I-" + label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids), dtype=torch.long, device=device
    )
    batch_labels = torch.tensor(
        sequence_padding(batch_labels), dtype=torch.long, device=device
    )
    return batch_token_ids, batch_labels

def trans_entity2tuple(scores):
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith("B-"):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (
                (len(entity_ids[-1]) > 0)
                and flag_tag.startswith("I-")
                and (flag_tag[2:] == entity_ids[-1][-1])
            ):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids

def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for token_ids, label in tqdm(data):
        scores = model.predict(token_ids)
        attention_mask = label.gt(0)

        # token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2

class Evaluator(Callback):
    def __init__(self):
        self.best_val_f1 = 0.0

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
        print(f"[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}")
        print(
            f"[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n"
        )


if __name__ == "__main__":

    args = parse_args()

    model_dir = args.model_dir
    data_dir = args.datasets_dir

    config_path = os.path.join(model_dir, "config.json")
    checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
    dict_path = os.path.join(model_dir, "vocab.txt")

    if not os.path.isfile(checkpoint_path):
        print("cant found checkpoint_path: {}".format(checkpoint_path))
        assert os.path.isfile(checkpoint_path)


    if not os.path.isfile(config_path):
        print("cant found config path: {}".format(config_path))
        assert os.path.isfile(config_path)


    if not os.path.isfile(dict_path):
        print("cant found dict_path: {}".format(dict_path))
        assert os.path.isfile(dict_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(42)

    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    
    train_data_file = os.path.join(data_dir, "example.train")
    dev_data_file = os.path.join(data_dir, "example.dev")

    if not os.path.isfile(train_data_file):
        print("cant found train data file: {}".format(train_data_file))
        assert os.path.isfile(train_data_file)

    if not os.path.isfile(dev_data_file):
        print("cant found dev data file: {}".format(dev_data_file))
        assert os.path.isfile(dev_data_file)

    train_dataloader = DataLoader(
        MyDataset(train_data_file),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    valid_dataloader = DataLoader(
        MyDataset(dev_data_file), batch_size=batch_size, collate_fn=collate_fn
    )

    quant_training_args = quan_train_config()
    quant_model_args = quan_model_config()
    quant_bert_args = quant_bert_config()
    with open(config_path, "r") as f:
        data = json.load(f)
        quant_bert_args.__dict__.update(data)

    model = Model()
    inject_ls_layer(model, quant_training_args, quant_model_args, quant_bert_args)

    model.to(device)

    model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))

    evaluator = Evaluator()

    model.fit(train_dataloader, epochs=3, steps_per_epoch=None, callbacks=[evaluator])

    quant_dir = "quant_base/"
    
    if not os.path.isdir(quant_dir):
        os.makedirs(quant_dir)
    save_file = os.path.join(quant_dir, "pytorch_model.bin")
    
    model.save_weights(save_file)
