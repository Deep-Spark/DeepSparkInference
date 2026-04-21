#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import paddle
import tensorrt
from tensorrt import Dims
from cuda import cuda, cudart

from transform import Transformation
from dataset import LibriSpeech
from decoder import CTCDecoder

from utils import VOCABLIST as vocab_list
from utils.error_rate import wer
from utils import deepspeech2_trtapi_ixrt, setup_io_bindings
from load_ixrt_plugin import load_ixrt_plugin

load_ixrt_plugin()

def parse_config():
    parser = argparse.ArgumentParser(description="The DeepSpeech2 network Inference on LibriSpeech dataset.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="DeepSpeech2",
        help="The speech recognition model(DeepSpeech2)",
    )
    parser.add_argument(
        "--preprocess_config",
        type=str,
        default="data/preprocess.yaml",
        help="The preprocess input file",
    )
    parser.add_argument(
        "--engine_file",
        type=str,
        default="../../../../../data/checkpoints/deepspeech2/deepspeech2.engine",
        help="engine file path"
    )
    parser.add_argument(
        "--decoder_file",
        type=str,
        default="../../../../../data/checkpoints/deepspeech2/decoder.pdparams",
        help="ctcdecoder checkpoints file"
    )
    parser.add_argument(
        "--lang_model_path",
        type=str,
        default="../../../../../data/checkpoints/deepspeech2/lm/common_crawl_00.prune01111.trie.klm",
        help="The language model path"
    )
    # dataset
    parser.add_argument(
        '--dataroot',
        default="../../../../../data/datasets/LibriSpeech",
        help='location to download dataset(s)'
    )
    parser.add_argument("--bsz", type=int, default=1, help="Dynamic input")
    parser.add_argument("--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--wer_target", type=float, default=-1.0)
    parser.add_argument("--test_num_samples", type=int, default=-1)

    config = parser.parse_args()
    return config


def test_result(data, engine, context, decoder, test_num_samples):

    input_name = "input"
    output_name = "output"

    data_len = len(data)
    wer_sum = 0.0

    if test_num_samples != -1:
        data_len = test_num_samples

    for i in tqdm(range(data_len), desc="Testing WER"):

        start_time = time.time()
        audio, text = data[i]
        audio_shape = audio.shape
        # print(f"audio_shape: {audio_shape}")

        # Set the input shape
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims(audio_shape))

        inputs, outputs, allocations = setup_io_bindings(engine, context)
        pred_output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], audio, audio.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)

        if config.use_async:
            stream = cuda.Stream()
            context.execute_async_v2(allocations, stream.handle)
            stream.synchronize()
        else:
            context.execute_v2(allocations)

        err, = cuda.cuMemcpyDtoH(pred_output, outputs[0]["allocation"], outputs[0]["nbytes"])
        assert(err == cuda.CUresult.CUDA_SUCCESS)

        eouts = paddle.to_tensor(pred_output)
        eouts_len = paddle.to_tensor([eouts.shape[1]])
        probs = decoder.softmax(eouts)
        batch_size = probs.shape[0]

        decoder.init_decoder(
                batch_size,
                vocab_list,
                "ctc_beam_search",
                config.lang_model_path,
                1.9,
                0.3,
                500,
                1.0,
                40,
                8
        )
        decoder.reset_decoder(batch_size=batch_size)
        decoder.next(probs, eouts_len)
        trans_best, trans_beam = decoder.decode()
        # print(f"result_transcripts: {trans_best}")
        # print(f"text: {text}")
        cur_wer = wer(text, trans_best[0], True)
        print(f"wer: {cur_wer}")
        wer_sum += cur_wer

    wer_avg = wer_sum / data_len
    print(f"wer_avg: {wer_avg}")
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["wer_avg"] = round(wer_avg, 3)
    print(metricResult)
    return wer_avg


def main(config):

    # Step1:build dataset
    preprocessing = Transformation(config.preprocess_config)
    dataset = LibriSpeech(config.dataroot, preprocessing)

    # Step2: load engine
    engine, context = deepspeech2_trtapi_ixrt(config.engine_file)

    # Step3: load decoder
    decoder = CTCDecoder(
            odim=31,
            enc_n_units=2048,
            blank_id=0,
            dropout_rate=0.0,
            reduction=True,
            batch_average=True,
            grad_norm_type=None
    )
    decoder_state_dict = paddle.load(config.decoder_file)
    decoder.set_state_dict(decoder_state_dict)

    # Step4: run test
    wer = test_result(dataset, engine, context, decoder, config.test_num_samples)
    status = 'Pass' if wer <= config.wer_target else 'Fail'

    print("="*30)
    print(f"\nCheck AUC:     Test : {wer}    Target:{config.wer_target}   State : {status}")
    print("="*30)


if __name__ == "__main__":
    config = parse_config()
    main(config)
