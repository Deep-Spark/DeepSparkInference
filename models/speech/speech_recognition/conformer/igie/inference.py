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

from __future__ import print_function

import argparse
import copy
import logging 
logging.basicConfig(level=logging.INFO, format = '[%(asctime)s %(filename)s line:%(lineno)d] %(levelname)s: %(message)s')
logging.getLogger('autotvm').setLevel(logging.ERROR)
logging.getLogger('strategy').setLevel(logging.ERROR)
logging.getLogger('te_compiler').setLevel(logging.ERROR)

import sys

from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import multiprocessing
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import compute_cer

from wenet.dataset.dataset import Dataset
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config
try:
    from swig_decoders import map_batch
except ImportError:
    print('Please install ctc decoders first by refering to\n' +
          'https://github.com/Slyne/ctc_decoder.git')
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--engine', required=True, help='igie engine path.')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type', default='raw', choices=['raw', 'shard'], help='train and cv data type')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--encoder', required=False, help='encoder magicmind model')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--label', required=True, help='label file path')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch size.')
    parser.add_argument('--seq_len', type=int, default=384, help='inference seq length.')
    parser.add_argument("--input_name", 
                        type=str,
                        nargs="+", 
                        required=True, 
                        help="input name of the model.")
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search', 
                            'ctc_prefix_beam_search',
                            'attention_rescoring'
                        ],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of workers used in pytorch dataloader.")
    parser.add_argument("--warmup", 
                        type=int, 
                        default=3, 
                        help="number of warmup before test.")  
    parser.add_argument('--fps_target',
                        type=float,
                        default=0.0)
    parser.add_argument('--acc_target',
                        type=float,
                        default=0.0)

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pprint(vars(args), indent=2)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = True
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=args.num_workers)

    # Load dict
    vocabulary = []
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    

    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
    device = tvm.device(target.kind.name, 0)
    
    lib = tvm.runtime.load_module(args.engine)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))
    
    if args.perf_only:
        ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)        
        prof_res = np.array(ftimer().results) * 1000 
        fps = args.batch_size * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
    else:
        # warm up
        for _ in range(args.warmup):
            module.run()
            
        with open(args.result_file, 'w') as fout:
            for _, batch in enumerate(test_data_loader):
                keys, feats, label, feats_lengths, label_lengths = batch
                feats, feats_lengths = feats.numpy(), feats_lengths.numpy()
                seq_len = feats.shape[1]
                if seq_len > args.seq_len:
                    continue

                if feats.shape[0] == args.batch_size:

                    speech_data = tvm.nd.array(feats, device)
                    speech_lengths_data = tvm.nd.array([feats_lengths], device)
                    module.set_input("speech", speech_data)
                    module.set_input("speech_lengths", speech_lengths_data)
                    
                    module.run()
                    
                    encoder_out, encoder_out_lens, ctc_log_probs = module.get_output(0).asnumpy(), module.get_output(1).asnumpy(), module.get_output(2).asnumpy()

                    preds = torch.from_numpy(ctc_log_probs)
                    beam_log_probs, beam_log_probs_idx = torch.topk(preds, k=4, dim=2)
                    
                    encoder_out = np.array(encoder_out, dtype="float32")
                    encoder_out_lens = np.array(encoder_out_lens, dtype="int32")
                    ctc_log_probs = np.array(ctc_log_probs, dtype="float32")
                    beam_log_probs = np.array(beam_log_probs, dtype="float32")
                    beam_log_probs_idx = np.array(beam_log_probs_idx, dtype="int64")

                    beam_size = beam_log_probs.shape[-1]
                    batch_size = beam_log_probs.shape[0]
                    num_processes = min(multiprocessing.cpu_count(), batch_size)
                    if args.mode == 'ctc_greedy_search':
                        if beam_size != 1:
                            log_probs_idx = beam_log_probs_idx[:, :, 0]
                        batch_sents = []
                        for idx, seq in enumerate(log_probs_idx):
                            batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
                        
                        hyps = map_batch(batch_sents, vocabulary, num_processes, True, 0)
                
                    for i, key in enumerate(keys):
                        content = hyps[i]
                        fout.write('{} {}\n'.format(key, content))

        Acc = compute_cer.get_acc(args.label, args.result_file)
        print(f"* Accuracy: {Acc} %")

if __name__ == '__main__':
    main()
