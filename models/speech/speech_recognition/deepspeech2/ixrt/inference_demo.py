#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import json
import random
import argparse
import soundfile
import numpy as np
from tqdm import tqdm

import torch
import paddle
import tensorrt
from tensorrt import Dims
from cuda import cuda, cudart

from transform import Transformation
from decoder import CTCDecoder

from utils import VOCABLIST as vocab_list
from utils import deepspeech2_trtapi_ixrt, setup_io_bindings

from load_ixrt_plugin import load_ixrt_plugin

load_ixrt_plugin()

def parse_config():
    parser = argparse.ArgumentParser(description="The DeepSpeech2 network Inference demo and performance.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="DeepSpeech2",
        help="The speech recognition model(DeepSpeech2)",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default="data/demo_002_en.wav",
        help="The input speech wave",
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
    parser.add_argument("--bsz", type=int, default=1, help="Dynamic input")
    parser.add_argument("--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--run_loop", type=int, default=-1)
    parser.add_argument("--warm_up", type=int, default=-1)
    parser.add_argument("--throughput_target", type=float, default=-1.0)

    config = parser.parse_args()
    return config


def main(config):
    # Step1: Load the input wave
    assert os.path.isfile(config.audio_file), "The input audio file must be existed!"
    audio, sample_rate = soundfile.read(config.audio_file, dtype="int16", always_2d=True)
    audio = audio[:, 0]
    print(f"audio shape: {audio.shape}")

    # fbank
    preprocess_args = {"train": False}
    preprocessing = Transformation(config.preprocess_config)
    input_data = preprocessing(audio, **preprocess_args)
    input_data = np.expand_dims(input_data.astype(np.float32), axis=0)
    print(f"feat shape: {input_data.shape}")

    # Step2: Load the engine
    engine, context = deepspeech2_trtapi_ixrt(config.engine_file)

    input_shape = input_data.shape
    print("input shape: ", input_shape)

    input_idx = engine.get_binding_index("input")
    context.set_binding_shape(input_idx, Dims(input_shape))

    inputs, outputs, allocations = setup_io_bindings(engine, context)
    pred_output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

    err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], input_data, input_data.nbytes)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    print("\n Warm Up Start.")
    for i in range(config.warm_up): context.execute_v2(allocations)
    print("Warm Up Done.")

    run_times = []
    for i in range(config.run_loop):
        start_time = time.time()
        context.execute_v2(allocations)
        end_time = time.time()
        run_times.append(end_time - start_time)

    run_times.remove(max(run_times))
    run_times.remove(min(run_times))

    avg_time = sum(run_times) / len(run_times)
    throughput = pred_output.shape[1] / avg_time
    print(f"Executing {config.run_loop} done, Time: {avg_time}, ThroughPut: {throughput}")

    err, = cuda.cuMemcpyDtoH(pred_output, outputs[0]["allocation"], outputs[0]["nbytes"])
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Step3: Load the CTCDecoder
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
    print("result_transcripts: ", trans_best)

    status = 'Pass' if throughput >= config.throughput_target else 'Fail'

    print("="*30)
    print(f"\nCheck ThroughPut:     Test : {throughput}    Target:{config.throughput_target}   State : {status}")
    print("="*30)

    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["ThroughPut"] = round(throughput, 3)
    print(metricResult)


if __name__ == "__main__":
    config = parse_config()
    main(config)
