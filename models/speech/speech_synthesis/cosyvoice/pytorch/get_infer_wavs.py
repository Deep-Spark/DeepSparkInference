# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import argparse
import os
import sys
from tqdm import tqdm
import torchaudio
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2

def parse_args():
    parser = argparse.ArgumentParser(description='Extract CosyVoice2 inference embeddings.')
    parser.add_argument('--inference_dir', default='', type=str, help='The root path inference eval')
    parser.add_argument('--input_text', default='', type=str, help='The text required for inference wavs')
    parser.add_argument('--prompt_text', default='', type=str, help='The text required for prompt wavs')
    parser.add_argument('--prompt_wav_scp', default='', type=str, help='The path of prompt wavs')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--output_dir', default='', type=str, help='Output directory for inference wavs')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    input_texts = {}
    for line in tqdm(open(args.input_text, 'r', encoding='utf-8').readlines()):
        utt_i, infer_text = line.strip().split(maxsplit=1)
        input_texts[utt_i] = infer_text
    prompt_texts = {}
    for line in tqdm(open(args.prompt_text, 'r', encoding='utf-8').readlines()):
        utt_p, pro_text = line.strip().split(maxsplit=1)
        prompt_texts[utt_p] = pro_text
    prompt_wavs = {}
    for line in tqdm(open(args.prompt_wav_scp, 'r', encoding='utf-8').readlines()):
        utt, prompt_wav = line.strip().split(maxsplit=1)
        prompt_wavs[utt] = os.path.join(args.inference_dir, prompt_wav)

    # inference output wavs
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=args.fp16)
    for uttid in tqdm(input_texts.keys()):
        for i, j in enumerate(cosyvoice.inference_zero_shot(input_texts[uttid], prompt_texts[uttid], prompt_wavs[uttid], stream=False)):
            wav_name = os.path.join(args.output_dir, f"{uttid}.wav")
            torchaudio.save(wav_name, j['tts_speech'], cosyvoice.sample_rate)

    print(f"Inference results have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()