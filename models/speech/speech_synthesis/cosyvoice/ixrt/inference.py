#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import numpy as np

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import soundfile as sf
from pystoi import stoi

def main(config):
    use_fp16 = False
    if config.precision == "float16":
        use_fp16 = True
    cosyvoice = CosyVoice2(config.model_dir, load_jit=False, load_trt=True, load_vllm=False, fp16=use_fp16)

    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

    start_time = time.time()
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}_ixrt_fp16.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    end_time = time.time()
    forward_time = end_time - start_time

    print("execute time {} s:".format(forward_time))

    ref, sr0 = sf.read('./asset/zero_shot_reference.wav')
    deg, sr1 = sf.read('zero_shot_0_ixrt_fp16.wav')
    if sr0 != sr1:
        print('采样率错误')
        exit(1)

    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]
    stoi_score = stoi(ref, deg, sr0, extended=False)
    if stoi_score < config.stoi_target:
        print('精度异常')
        exit(1)
    print('stoi_score:',stoi_score)
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["execute_time"] = forward_time
    metricResult["metricResult"]["stoi_score"] = stoi_score
    print(metricResult)
    exit()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float16",
            help="The precision of datatype")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="model dir path",
    )
    parser.add_argument("--warm_up", type=int, default=3, help="warm_up count")
    parser.add_argument("--loop_count", type=int, default=5, help="loop count")
    parser.add_argument("--stoi_target", type=float, default=0.8, help="target mAP")

    config = parser.parse_args()
    print("config:", config)
    return config

if __name__ == "__main__":
    config = parse_config()
    main(config)