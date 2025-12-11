import json
import os
import numpy as np
import argparse
import time

import tensorrt
from tensorrt import Dims
from common import create_engine_context, get_io_bindings, setup_io_bindings

import sys
sys.path.append("../")
from plugin.transformer_cfg import TransformerBaseConfig
from plugin.trt import T5TRTDecoder, T5TRTEncoder,inference,benchmark

import torch
from torch.utils.data import DataLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, index):
        input = self.inputs[index]        
        return input
    
    def __len__(self):
        return len(self.inputs)





def generate_batch(features):
    all_inputs = []
    tmp = []
    for data in features:
        if len(tmp) == args.max_batch_size:
            batch_max_len = max([len(i) for i in tmp])
            new_tmp = []
            for i in tmp:
                i = i[:args.max_seq_len]
                i = [pad_id]*(batch_max_len-len(i)) + i
                new_tmp.append(i)
            all_inputs.append(np.array(new_tmp).astype(np.int32))
            tmp = []
        tmp.append(data)
        
    return all_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="build ixrt graph and convert weights", usage=""
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        required=True,
        help="max batch size for inference",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=102,
        help="max sequence length for inference",
    )
    parser.add_argument(
        "--data_dir",
        type=str
    )
    parser.add_argument(
        "--model_dir",
        type=str
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.max_seq_len <= 102
    pad_id = 1
    feature_file = os.path.join(args.data_dir,'features.json')

    with open(feature_file,'r') as f:
        features = json.loads(f.read())

    all_inputs = generate_batch(features)
    print(f"max_batch_size: {args.max_batch_size}, max_seq_len: {args.max_seq_len}")

    print("1. build engine")
    
    
    batch_size = args.max_batch_size
    config_path = os.path.join(args.model_dir,'transformer_config.json')
    config = TransformerBaseConfig(config_path)
    
    encoder_engine =  os.path.join(args.model_dir,'Encoder.engine')
    print(f"2 load encoder engine from {encoder_engine}") 
    encoder = T5TRTEncoder(encoder_engine,config, batch_size=batch_size) 
    
    
    decoder_engine =  os.path.join(args.model_dir,'Decoder.engine')
    print(f"3 load decoder_engine engine from {decoder_engine}") 
    decoder = T5TRTDecoder(decoder_engine,config,batch_size=batch_size)  
    
    
    device = torch.device("cuda:0")
    dataset = CustomDataset(all_inputs)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
    
    prev_tokens = torch.full((batch_size,1), int(config.sos_token_id),dtype = torch.int32).cuda()
    for i, data in enumerate(dataloader):     
        data = torch.squeeze(data,0).to(device)
        benchmark(config,encoder,decoder,data,prev_tokens)

    print("3. inference")
    
    total_time = 0
    
    num_sentences = 0
    for i, data in enumerate(dataloader):
        data = torch.squeeze(data,0).to(device)
        num_sentences += data.shape[0]
        start_time = time.time()
        benchmark(config,encoder,decoder,data,prev_tokens)
        end_time = time.time()
        total_time +=(end_time-start_time)
        
    QPS = num_sentences/total_time
    print(f"Translated {num_sentences} sentences, {QPS} sentences/s")
    target_qps = float(os.environ['Accuracy'])
    decoder.clear()
    encoder.clear() 

    print("QPS: = ", QPS, "target QPS: ", target_qps)
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["QPS"] = round(QPS, 3)
    print(metricResult)
    if QPS >= target_qps:
        print("pass!")
        exit()
    else:
        print("failed!")
        exit(10)
